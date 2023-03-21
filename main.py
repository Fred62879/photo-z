
if __name__ == '__main__':

    # from parsers.redshift_parser import parse_args
    from parsers.imagenet_parser import parse_args
    from utils import get_dino_pipeline, get_redshift_dataset, get_redshift_trainer, get_optimizer, get_imagenet_dataset

    args, args_str = parse_args()
    kwargs = vars(args)

    dino_model = get_dino_pipeline(**kwargs)
    # dataset = get_redshift_dataset(**kwargs)
    dataset = get_imagenet_dataset(**kwargs)
    optim_cls, optim_params = get_optimizer(**kwargs)
    trainer = get_redshift_trainer(dino_model, dataset, None, optim_cls, optim_params, "pre_training", **kwargs)

    if kwargs['trainer_mode'] == "pre_training":
        trainer.set_mode("pre_training")
        trainer.train()
    elif kwargs['trainer_mode'] == "redshif_est":
        trainer.set_mode("redshift_est")
        trainer.train()
    elif kwargs['trainer_mode'] == "validate":
        trainer.validate()
