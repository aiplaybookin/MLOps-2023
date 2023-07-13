import hydra
from omegaconf import DictConfig, OmegaConf
from pprint import pprint

@hydra.main(version_base=None, config_path='configs', config_name='config') # configs is folder, config is yaml file
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    pprint(OmegaConf.to_object(cfg))
    #trainer = hydra.utils.instantiate(cfg.trainer)

    #print(f"{trainer=}")

if __name__ == "__main__":
    main()