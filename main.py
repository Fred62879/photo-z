
if __name__ == "__main__":

    from parsers.redshift_parser import parse_args
    # from parsers.imagenet_parser import parse_args
    from utils.get_from_config import get_pipeline, get_redshift_dataset, get_redshift_trainer, get_optimizer, get_imagenet_dataset, get_imagenet_trainer

    args, args_str = parse_args()
    kwargs = vars(args)

    model = get_pipeline(**kwargs)
    dataset = get_redshift_dataset(**kwargs)
    # dataset = get_imagenet_dataset(**kwargs)
    optim_cls, optim_params = get_optimizer(**kwargs)

    if kwargs["trainer_mode"] == "pre_training":
        # trainer = get_imagenet_trainer(
        trainer = get_redshift_trainer(
            model, dataset[0], optim_cls, optim_params, "pre_training", **kwargs)
        trainer.train()

    elif kwargs["trainer_mode"] == "redshift_train":
        trainer = get_redshift_trainer(
            model, dataset[:2], optim_cls, optim_params, "redshift_train", **kwargs)
        trainer.train()

    elif kwargs["trainer_mode"] == "test":
        trainer = get_redshift_trainer(
            model, dataset[2], optim_cls, optim_params, "redshift_est", **kwargs)
        trainer.validate(mode="test")
