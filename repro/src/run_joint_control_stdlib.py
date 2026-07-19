#!/usr/bin/env python3
"""Fully-trainable CPU test of CAffNet joint optimization (Eq. 4).

This reduced safety-control task uses an input-dependent rotating affine face
n(x)^T y = b(x), equivalent to two one-sided inequalities. Fully trainable
one-hidden-layer tanh networks produce the unconstrained output f_theta and
null-space choice w_phi. The comparison includes:

1. jointly trained CAffNet f_theta + w_phi;
2. unconstrained training followed by a-posteriori orthogonal projection;
3. orthogonal projection trained end-to-end with w=0.

The task loss is an anisotropic quadratic control-tracking objective with a
closed-form constrained oracle. It is an independent reduced mechanism test,
not the paper's unreleased learned-optimizer or robot-control benchmark.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import statistics
import sys
import time


def zeros_like(params):
    return [[[0.0 for _ in row] for row in matrix] if isinstance(matrix[0], list) else [0.0 for _ in matrix] for matrix in params]


class MLP:
    """Fully trainable 2 -> hidden -> 2 tanh MLP with Adam."""

    def __init__(self, rng, hidden):
        scale1 = 1.0 / math.sqrt(2.0)
        scale2 = 1.0 / math.sqrt(hidden)
        self.params = [
            [[rng.gauss(0.0, scale1) for _ in range(2)] for _ in range(hidden)],
            [0.0 for _ in range(hidden)],
            [[rng.gauss(0.0, scale2) for _ in range(hidden)] for _ in range(2)],
            [0.0, 0.0],
        ]
        self.m = zeros_like(self.params)
        self.v = zeros_like(self.params)
        self.step_number = 0

    def forward(self, xs):
        w1, b1, w2, b2 = self.params
        outputs = []
        caches = []
        for x in xs:
            inp = [x, x * x]
            hidden = [math.tanh(sum(weight * value for weight, value in zip(row, inp)) + bias) for row, bias in zip(w1, b1)]
            out = [sum(weight * value for weight, value in zip(row, hidden)) + bias for row, bias in zip(w2, b2)]
            outputs.append(out)
            caches.append((inp, hidden))
        return outputs, caches

    def backward(self, caches, dout):
        w1, _, w2, _ = self.params
        gw1 = [[0.0, 0.0] for _ in w1]
        gb1 = [0.0 for _ in w1]
        gw2 = [[0.0 for _ in row] for row in w2]
        gb2 = [0.0, 0.0]
        for (inp, hidden), grad_out in zip(caches, dout):
            for out_index in range(2):
                gb2[out_index] += grad_out[out_index]
                for hidden_index, value in enumerate(hidden):
                    gw2[out_index][hidden_index] += grad_out[out_index] * value
            grad_hidden = [sum(w2[out_index][hidden_index] * grad_out[out_index] for out_index in range(2)) for hidden_index in range(len(hidden))]
            grad_pre = [grad * (1.0 - value * value) for grad, value in zip(grad_hidden, hidden)]
            for hidden_index, grad in enumerate(grad_pre):
                gb1[hidden_index] += grad
                gw1[hidden_index][0] += grad * inp[0]
                gw1[hidden_index][1] += grad * inp[1]
        return [gw1, gb1, gw2, gb2]

    def gradient_norms(self, grads):
        hidden = math.sqrt(sum(value * value for row in grads[0] for value in row) + sum(value * value for value in grads[1]))
        output = math.sqrt(sum(value * value for row in grads[2] for value in row) + sum(value * value for value in grads[3]))
        return {"hidden": hidden, "output": output, "total": math.sqrt(hidden * hidden + output * output)}

    def adam_step(self, grads, lr):
        self.step_number += 1
        beta1 = 0.9
        beta2 = 0.999
        for group_index, (params, group_grads) in enumerate(zip(self.params, grads)):
            if params and isinstance(params[0], list):
                for row_index, (row, grad_row) in enumerate(zip(params, group_grads)):
                    for col_index, grad in enumerate(grad_row):
                        self.m[group_index][row_index][col_index] = beta1 * self.m[group_index][row_index][col_index] + (1.0 - beta1) * grad
                        self.v[group_index][row_index][col_index] = beta2 * self.v[group_index][row_index][col_index] + (1.0 - beta2) * grad * grad
                        mhat = self.m[group_index][row_index][col_index] / (1.0 - beta1**self.step_number)
                        vhat = self.v[group_index][row_index][col_index] / (1.0 - beta2**self.step_number)
                        row[col_index] -= lr * mhat / (math.sqrt(vhat) + 1e-8)
            else:
                for index, grad in enumerate(group_grads):
                    self.m[group_index][index] = beta1 * self.m[group_index][index] + (1.0 - beta1) * grad
                    self.v[group_index][index] = beta2 * self.v[group_index][index] + (1.0 - beta2) * grad * grad
                    mhat = self.m[group_index][index] / (1.0 - beta1**self.step_number)
                    vhat = self.v[group_index][index] / (1.0 - beta2**self.step_number)
                    params[index] -= lr * mhat / (math.sqrt(vhat) + 1e-8)


def control_problem(xs):
    normals = []
    bounds = []
    targets = []
    qdiag = []
    for x in xs:
        angle = 0.85 * math.sin(0.75 * math.pi * x) + 0.25 * x
        normal = [math.cos(angle), math.sin(angle)]
        bound = 0.18 * math.sin(2.0 * math.pi * x)
        target = [1.0 + 0.45 * math.sin(math.pi * x), -0.35 + 0.4 * math.cos(1.5 * math.pi * x)]
        weights = [1.0 + 1.5 * (x > 0.0), 7.0 - 2.0 * math.cos(math.pi * x)]
        normals.append(normal)
        bounds.append(bound)
        targets.append(target)
        qdiag.append(weights)
    return normals, bounds, targets, qdiag


def caffine(outputs, free_vectors, normals, bounds):
    result = []
    for output, free, normal, bound in zip(outputs, free_vectors, normals, bounds):
        violation = normal[0] * output[0] + normal[1] * output[1] - bound
        tangent = [-normal[1], normal[0]]
        tangent_amount = tangent[0] * free[0] + tangent[1] * free[1]
        result.append([
            output[0] - normal[0] * violation + tangent[0] * tangent_amount,
            output[1] - normal[1] * violation + tangent[1] * tangent_amount,
        ])
    return result


def orthogonal(outputs, normals, bounds):
    return caffine(outputs, [[0.0, 0.0] for _ in outputs], normals, bounds)


def constrained_oracle(targets, normals, bounds, qdiag):
    result = []
    for target, normal, bound, weights in zip(targets, normals, bounds, qdiag):
        denominator = normal[0] * normal[0] / weights[0] + normal[1] * normal[1] / weights[1]
        multiplier = (normal[0] * target[0] + normal[1] * target[1] - bound) / denominator
        result.append([
            target[0] - normal[0] * multiplier / weights[0],
            target[1] - normal[1] * multiplier / weights[1],
        ])
    return result


def objective(outputs, targets, qdiag):
    return sum(
        0.5 * (weights[0] * (output[0] - target[0]) ** 2 + weights[1] * (output[1] - target[1]) ** 2)
        for output, target, weights in zip(outputs, targets, qdiag)
    ) / len(outputs)


def output_gradient(outputs, targets, qdiag):
    count = len(outputs)
    return [[
        weights[0] * (output[0] - target[0]) / count,
        weights[1] * (output[1] - target[1]) / count,
    ] for output, target, weights in zip(outputs, targets, qdiag)]


def tangent_gradient(gradients, normals):
    result = []
    for gradient, normal in zip(gradients, normals):
        dot = normal[0] * gradient[0] + normal[1] * gradient[1]
        result.append([gradient[0] - normal[0] * dot, gradient[1] - normal[1] * dot])
    return result


def max_residual(outputs, normals, bounds):
    return max(abs(normal[0] * output[0] + normal[1] * output[1] - bound) for output, normal, bound in zip(outputs, normals, bounds))


def mse(outputs, targets):
    return sum((output[j] - target[j]) ** 2 for output, target in zip(outputs, targets) for j in range(2)) / (2 * len(outputs))


def parameter_count(network):
    return sum(len(row) for row in network.params[0]) + len(network.params[1]) + sum(len(row) for row in network.params[2]) + len(network.params[3])


def run_seed(seed, steps, hidden, lr, train_points, test_points):
    rng = random.Random(seed)
    train_x = [-1.0 + 2.0 * index / (train_points - 1) for index in range(train_points)]
    test_x = [-1.0 + 2.0 * index / (test_points - 1) for index in range(test_points)]
    train_problem = control_problem(train_x)
    test_problem = control_problem(test_x)
    train_normals, train_bounds, train_targets, train_qdiag = train_problem
    test_normals, test_bounds, test_targets, test_qdiag = test_problem
    oracle_test = constrained_oracle(test_targets, test_normals, test_bounds, test_qdiag)

    f_joint = MLP(rng, hidden)
    w_joint = MLP(rng, hidden)
    f_posthoc = MLP(rng, hidden)
    f_fixed = MLP(rng, hidden)
    curves = []
    initial_gradients = None

    for step in range(steps + 1):
        f_values, f_cache = f_joint.forward(train_x)
        w_values, w_cache = w_joint.forward(train_x)
        joint = caffine(f_values, w_values, train_normals, train_bounds)
        joint_tangent_grad = tangent_gradient(output_gradient(joint, train_targets, train_qdiag), train_normals)
        f_joint_grads = f_joint.backward(f_cache, joint_tangent_grad)
        w_joint_grads = w_joint.backward(w_cache, joint_tangent_grad)

        unconstrained, posthoc_cache = f_posthoc.forward(train_x)
        posthoc_grads = f_posthoc.backward(posthoc_cache, output_gradient(unconstrained, train_targets, train_qdiag))

        fixed_raw, fixed_cache = f_fixed.forward(train_x)
        fixed = orthogonal(fixed_raw, train_normals, train_bounds)
        fixed_grads = f_fixed.backward(fixed_cache, tangent_gradient(output_gradient(fixed, train_targets, train_qdiag), train_normals))

        if step == 0:
            initial_gradients = {
                "theta": f_joint.gradient_norms(f_joint_grads),
                "phi": w_joint.gradient_norms(w_joint_grads),
                "posthoc_theta": f_posthoc.gradient_norms(posthoc_grads),
                "fixed_in_loop_theta": f_fixed.gradient_norms(fixed_grads),
            }
        if step % max(1, steps // 20) == 0 or step == steps:
            curves.append({
                "seed": seed,
                "step": step,
                "joint_objective": objective(joint, train_targets, train_qdiag),
                "posthoc_unconstrained_objective": objective(unconstrained, train_targets, train_qdiag),
                "fixed_in_loop_objective": objective(fixed, train_targets, train_qdiag),
                "joint_constraint_residual": max_residual(joint, train_normals, train_bounds),
                "unconstrained_constraint_residual": max_residual(unconstrained, train_normals, train_bounds),
            })
        if step == steps:
            break
        f_joint.adam_step(f_joint_grads, lr)
        w_joint.adam_step(w_joint_grads, lr)
        f_posthoc.adam_step(posthoc_grads, lr)
        f_fixed.adam_step(fixed_grads, lr)

    f_values, _ = f_joint.forward(test_x)
    w_values, _ = w_joint.forward(test_x)
    joint = caffine(f_values, w_values, test_normals, test_bounds)
    joint_w_ablated = orthogonal(f_values, test_normals, test_bounds)
    unconstrained, _ = f_posthoc.forward(test_x)
    posthoc = orthogonal(unconstrained, test_normals, test_bounds)
    fixed_raw, _ = f_fixed.forward(test_x)
    fixed = orthogonal(fixed_raw, test_normals, test_bounds)
    oracle_objective = objective(oracle_test, test_targets, test_qdiag)

    def report(outputs):
        value = objective(outputs, test_targets, test_qdiag)
        return {
            "objective": value,
            "objective_gap": value - oracle_objective,
            "mse_to_oracle": mse(outputs, oracle_test),
            "max_constraint_residual": max_residual(outputs, test_normals, test_bounds),
        }

    return {
        "summary": {
            "seed": seed,
            "initial_gradient_norms": initial_gradients,
            "oracle_objective": oracle_objective,
            "joint": report(joint),
            "joint_w_ablated": report(joint_w_ablated) | {"mse_to_joint": mse(joint_w_ablated, joint)},
            "posthoc_fixed_orthogonal": report(posthoc) | {
                "unconstrained_objective_before_projection": objective(unconstrained, test_targets, test_qdiag),
                "unconstrained_max_constraint_residual": max_residual(unconstrained, test_normals, test_bounds),
            },
            "fixed_orthogonal_trained_in_loop": report(fixed),
        },
        "curves": curves,
    }


def stats(values):
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def aggregate(runs):
    return {
        "joint_objective_gap": stats([run["joint"]["objective_gap"] for run in runs]),
        "posthoc_objective_gap": stats([run["posthoc_fixed_orthogonal"]["objective_gap"] for run in runs]),
        "fixed_in_loop_objective_gap": stats([run["fixed_orthogonal_trained_in_loop"]["objective_gap"] for run in runs]),
        "posthoc_over_joint_objective_ratio": stats([run["posthoc_fixed_orthogonal"]["objective"] / run["joint"]["objective"] for run in runs]),
        "w_ablation_objective_increase": stats([run["joint_w_ablated"]["objective"] - run["joint"]["objective"] for run in runs]),
        "joint_max_constraint_residual": stats([run["joint"]["max_constraint_residual"] for run in runs]),
        "unconstrained_max_constraint_residual": stats([run["posthoc_fixed_orthogonal"]["unconstrained_max_constraint_residual"] for run in runs]),
        "theta_initial_hidden_gradient": stats([run["initial_gradient_norms"]["theta"]["hidden"] for run in runs]),
        "phi_initial_hidden_gradient": stats([run["initial_gradient_norms"]["phi"]["hidden"] for run in runs]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/joint_control")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--hidden", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--train-points", type=int, default=64)
    parser.add_argument("--test-points", type=int, default=501)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)
    started = time.perf_counter()
    results = [run_seed(seed, args.steps, args.hidden, args.lr, args.train_points, args.test_points) for seed in seeds]
    runtime = time.perf_counter() - started
    runs = [result["summary"] for result in results]
    network = MLP(random.Random(0), args.hidden)
    payload = {
        "experiment": "CAffNet fully-trainable joint optimization on rotating safety-control affine faces",
        "paper_anchor": "Section 3.2 Eq. (4) joint optimization; Section 4.3 a-posteriori projection comparison",
        "fidelity": {
            "matched": [
                "fully trainable hidden and output parameters",
                "separate jointly optimized f_theta and w_phi networks",
                "input-dependent affine constraints",
                "posthoc fixed projection comparison",
                "fixed projection trained end-to-end adversarial control",
                "multiple seeds and exact feasibility checks",
            ],
            "deviations": [
                "independent synthetic rotating safety boundary rather than unreleased robot/control data",
                "equality face represented as two one-sided inequalities rather than 11/13 paper inequalities",
                "one hidden layer of reduced width rather than paper's three 200-unit layers",
                "local CPU rather than V100",
            ],
        },
        "configuration": vars(args) | {
            "seeds": seeds,
            "trainable_parameters_per_network": parameter_count(network),
            "constraint": "n(x)^T y = 0.18 sin(2 pi x), encoded as two one-sided inequalities",
            "task": "anisotropic control tracking on an input-dependent rotating safety boundary",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "dependencies": "Python standard library only",
        },
        "runtime_seconds": runtime,
        "runs": runs,
        "aggregate": aggregate(runs),
        "interpretation_guardrail": "The posthoc baseline matches the paper's narrow comparison. The fixed-in-loop baseline tests and may refute the stronger claim that w_phi is uniquely necessary when f_theta is itself trained through the projection.",
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    curve_path = os.path.join(args.output_dir, "learning_curves.csv")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    rows = [row for result in results for row in result["curves"]]
    with open(curve_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2))
    print("wrote", summary_path)
    print("wrote", curve_path)


if __name__ == "__main__":
    main()
