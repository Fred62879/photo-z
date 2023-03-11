
if __name__ == '__main__':

    from parsers.protein_parser import parse_args
    from utils import get_protein_sdf_model, get_protein_dataset, get_protein_trainer, get_optimizer

    args, args_str = parse_args()
    kwargs = vars(args)

    #pipeline = get_protein_pipeline(**args)
    sdf_model = get_protein_sdf_model(**kwargs)
    dataset = get_protein_dataset(**kwargs)
    optim_cls, optim_params = get_optimizer(**kwargs)
    trainer = get_protein_trainer(sdf_model, dataset, None, optim_cls, optim_params, "train", **kwargs)

    if kwargs['trainer_mode'] == 'train':
        trainer.train()
    elif kwargs['trainer_mode'] == 'validate_mesh':
        trainer.set_mode("validate")
        threshs = [-0.001,-0.0025,-0.005,-0.01,-0.02,0.0,0.001,0.0025,0.005,0.01,0.02]
        for thresh in threshs:
            trainer.validate_mesh(resolution=256, threshold=thresh)
