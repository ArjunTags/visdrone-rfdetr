import argparse

from model import MODELS, build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="runs/rfdetr")
    ap.add_argument("--size", choices=MODELS, default="base")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-encoder", type=float, default=1.5e-4)
    ap.add_argument("--resolution", type=int, default=None,
                    help="input size; VisDrone objects are tiny, so larger can help (must satisfy the model's divisibility rule)")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2, help="use 0 if training hangs on Windows")
    ap.add_argument("--grad-checkpointing", action="store_true", help="save GPU memory, slower")
    ap.add_argument("--resume", default=None, help="path to checkpoint.pth to resume")
    args = ap.parse_args()

    model = build_model(args.size)

    kwargs = dict(
        dataset_dir=args.dataset,
        output_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        early_stopping=True,
        early_stopping_patience=args.patience,
        use_ema=True,
        tensorboard=False,
        num_workers=args.num_workers,
        gradient_checkpointing=args.grad_checkpointing,
    )
    if args.resolution:
        kwargs["resolution"] = args.resolution
    if args.resume:
        kwargs["resume"] = args.resume

    model.train(**kwargs)
    print(f"\nDone. Best weights: {args.out}/checkpoint_best_total.pth")


if __name__ == "__main__":
    main()
