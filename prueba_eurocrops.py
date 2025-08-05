import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from shapeft.datasets.eurocrops import EuroCrops


@hydra.main(version_base=None, config_path="configs/dataset/", config_name="eurocrops")
def main(cfg: DictConfig) -> None:
    dataset = instantiate(cfg, root_path="data/", split="test")
    print(dataset.__getitem__(1))


if __name__ == "__main__":
    main()
