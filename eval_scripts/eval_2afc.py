#!/usr/bin/env python
"""
2-AFC Recognition Memory Evaluation

Task: Study N images, then for each test trial show 2 images (original + foil)
      and ask which was in the study sequence.

Usage:
    python -m eval_scripts.eval_2afc --models gpt-4o --n-images 100 --n-trials 100 --foil-type novel
    python -m eval_scripts.eval_2afc --models gpt-4o --n-images 5  # pilot run
"""
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import re
from datetime import datetime
from tasks.recognition import AFCRecognitionTask
from evaluators.openai_evaluator import OpenAIEvaluator
from evaluators.anthropic_evaluator import AnthropicEvaluator
from evaluators.google_evaluator import GoogleEvaluator
from evaluators.qwen_evaluator import QwenEvaluator
from evaluators.molmo2_evaluator import Molmo2Evaluator
from src.metrics import calculate_2afc_metrics
from src.plotting import default_plots_dir, plot_2afc_all


def build_messages(evaluator, study_images, study_prompt, test_images, test_prompt):
    """Build API messages for 2-AFC trial with simulated acknowledgment."""

    # Study phase
    study_content = [{"type": "text", "text": study_prompt}]
    for img in study_images:
        study_content.append(evaluator._encode_image(img))

    # Test phase
    test_content = [{"type": "text", "text": test_prompt}]
    for img in test_images:
        test_content.append(evaluator._encode_image(img))

    return [
        {"role": "user", "content": study_content},
        {"role": "assistant", "content": "I have studied the sequence of images."},
        {"role": "user", "content": test_content}
    ]


def parse_response(text):
    """Parse 1 or 2 from response (digits or common paraphrases like 'first' / 'second')."""
    if text is None:
        return -1
    text = text.strip()
    if "1" in text and "2" not in text:
        return 1
    elif "2" in text and "1" not in text:
        return 2
    elif text.startswith("1"):
        return 1
    elif text.startswith("2"):
        return 2
    # Models often answer "The first image..." — no digit `1` in "first"
    lower = text.lower()
    has_first = re.search(r"\bfirst\b", lower) is not None
    has_second = re.search(r"\bsecond\b", lower) is not None
    if has_first and not has_second:
        return 1
    if has_second and not has_first:
        return 2
    return -1


def run_evaluation(evaluators, n_images=20, n_trials=None, foil_type='novel', dataset='things'):
    """Run 2-AFC evaluation on all evaluators.

    Each trial is an independent study+test episode: a fresh set of n_images is
    sampled, studied, then tested on one 2-AFC question. n_trials controls how
    many independent episodes are run.
    """
    n_trials = n_trials if n_trials is not None else n_images

    all_results = {}
    for evaluator in evaluators:
        print(f"\n=== {evaluator.get_name()} ===")

        print(f"  Probing capacity for {n_images} images...", end=" ", flush=True)
        if not evaluator.check_image_capacity(n_images + 2):
            print(f"SKIP — model rejected {n_images} images in a single request")
            continue
        print("OK")

        results = []

        for i in range(n_trials):
            task = AFCRecognitionTask(dataset_name=dataset, n_images=n_images, foil_type=foil_type)
            trial_data = task.get_trials()
            test_trial = trial_data['test_phase'][0]

            messages = build_messages(
                evaluator,
                trial_data['study_sequence'],
                trial_data['study_prompt'],
                test_trial['images'],
                test_trial['prompt'],
            )
            response_text = evaluator._call_api(messages)
            response = parse_response(response_text)
            correct = 1 if response == test_trial['target'] else 0

            results.append({
                'trial': i,
                'target': test_trial['target'],
                'response': response,
                'correct': correct,
                'foil_type': test_trial['type'],
                'raw_response': response_text
            })
            status = '✓' if correct else '✗'
            print(f"  Trial {i+1}/{n_trials}: {status}", end="\r")

        metrics = calculate_2afc_metrics(results)
        print(f"\n  Accuracy: {metrics['accuracy']:.1%} | d': {metrics['d_prime']:.2f} | Mem Score: {metrics['mem_score']:.2f}")

        all_results[evaluator.get_name()] = {
            'trials': results,
            **metrics
        }

    return all_results


def main():
    parser = argparse.ArgumentParser(description="2-AFC Recognition Evaluation")
    parser.add_argument("--models", nargs="+", default=["gpt-4o", "claude", "gemini"],
                        help="Models to evaluate: gpt-4o, claude, gemini, qwen, molmo2 (or provider model IDs)")
    parser.add_argument("--n-images", type=int, default=20,
                        help="Number of images in study sequence")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Number of test trials (default: same as n-images)")

    parser.add_argument("--foil-type", choices=["novel", "exemplar", "state", "all"], default="novel",
                        help="Type of foils to use")
    parser.add_argument("--dataset", choices=["things", "Brady2008", "cued-recall-imagenet"], default="things",
                        help="Dataset to use")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: results_2afc_<timestamp>.json)")
    parser.add_argument("--plot", action="store_true",
                        help="Write figures under output/plots (or --plot-dir)")
    parser.add_argument("--plot-dir", type=str, default=None,
                        help="Directory for figures (default: repo output/plots)")
    parser.add_argument("--max-image-size", type=int, default=512,
                        help="Max image dim (px) for local-inference evaluators (qwen, molmo2). Default 512.")
    parser.add_argument("--vision-chunk-size", type=int, default=None,
                        help="If set, qwen vision encoder runs on this many images per chunk (reduces peak GPU mem).")
    parser.add_argument("--attn", type=str, default=None,
                        choices=[None, "eager", "sdpa", "flash_attention_2"],
                        help="attn_implementation for local models (qwen). Default: HF default.")
    args = parser.parse_args()

    qwen_kwargs = dict(
        max_image_size=args.max_image_size,
        vision_chunk_size=args.vision_chunk_size,
        attn_implementation=args.attn,
    )

    evaluators = []
    for model in args.models:
        m = model.strip()
        if not m:
            continue
        if m == "gpt-4o":
            evaluators.append(OpenAIEvaluator("gpt-4o"))
        elif m == "claude":
            evaluators.append(AnthropicEvaluator())
        elif m == "gemini":
            evaluators.append(GoogleEvaluator())
        elif m == "qwen":
            evaluators.append(QwenEvaluator("Qwen/Qwen3-VL-8B-Instruct", **qwen_kwargs))
        elif m == "molmo2":
            evaluators.append(Molmo2Evaluator("allenai/Molmo2-8B"))
        elif m.startswith("claude"):
            evaluators.append(AnthropicEvaluator(m))
        elif m.startswith("gemini"):
            evaluators.append(GoogleEvaluator(m))
        elif m.startswith("qwen") or m.startswith("Qwen"):
            evaluators.append(QwenEvaluator(m, **qwen_kwargs))
        elif m.startswith("molmo") or m.startswith("allenai"):
            evaluators.append(Molmo2Evaluator(m))
        else:
            evaluators.append(OpenAIEvaluator(m))

    if not evaluators:
        print("No valid models specified. Use --models gpt-4o claude gemini qwen molmo2")
        return

    n_trials = args.n_trials if args.n_trials is not None else args.n_images

    print(f"Running 2-AFC evaluation:")
    print(f"  Models: {[e.get_name() for e in evaluators]}")
    print(f"  N images (study): {args.n_images}")
    print(f"  N trials (test): {n_trials}")

    print(f"  Foil type: {args.foil_type}")
    print(f"  Dataset: {args.dataset}")

    results = run_evaluation(
        evaluators,
        args.n_images,
        n_trials,
        args.foil_type,
        args.dataset,
    )

    if not results:
        print("No results produced — not writing output file.")
        return

    # Build output with metadata at the top
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_data = {
        "_metadata": {
            "task": "2-AFC Recognition",
            "timestamp": timestamp,
            "dataset": args.dataset,
            "n_images": args.n_images,
            "n_trials": n_trials,
            "models": [e.get_name() for e in evaluators],
            "summary": {
                model: {
                    "accuracy": results[model]["accuracy"],
                    "d_prime": results[model]["d_prime"],
                    "mem_score": results[model]["mem_score"],
                }
                for model in results
            },

            "foil_type": args.foil_type,
        },
        **results,
    }

    # Save results to results folder
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    if args.output:
        output_path = results_dir / args.output
    else:
        model_str = "+".join(e.get_name() for e in evaluators)
        output_path = results_dir / f"results_2afc_{model_str}_n{args.n_images}_{args.dataset}_{args.foil_type}.json"

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved to {output_path}")

    if args.plot:
        plot_dir = Path(args.plot_dir) if args.plot_dir else default_plots_dir()
        p_acc, p_met = plot_2afc_all(output_data, output_dir=plot_dir)
        print(f"Plots saved to {p_acc} and {p_met}")


if __name__ == "__main__":
    main()
