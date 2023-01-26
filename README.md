# Towards Better Graph Representation Learning with Parameterized Decomposition & Filtering

[![MIT License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

This is the code of the paper "Towards Better Graph Representation Learning with Parameterized Decomposition & Filtering".

## Requirements

The following packages need to be installed:

- `pytorch==1.13.0`
- `dgl==0.9.1`
- `ogb==1.3.5`
- `numpy`
- `easydict`
- `tensorboard`
- `tqdm`
- `json5`

## Usage

#### ZINC
- Change your current directory to [zinc](zinc);
- Configure hyper-parameters in [ZINC.json](zinc/ZINC.json);
- Run the script: `sh run_script.sh`.

#### ogbg-molpcba/ppa
- Change your current directory to [ogbg/mol](ogbg/mol) or [ogbg/ppa](ogbg/ppa);
- Configure hyper-parameters in [ogbg-molpcba.json](ogbg/mol/ogbg-molpcba.json) or [ogbg-ppa.json](ogbg/ppa/ogbg-ppa.json);
- Set dataset name in [run_script.sh](tu/run_script.sh);
- Run the script: `sh run_script.sh`.

#### TUDataset
- Change your current directory to [tu](tu);
- configure hyper-parameters in [configs/\<dataset\>.json](tu/configs);
- Set dataset name in [run_script.sh](tu/run_script.sh);
- Run the script: `sh run_script.sh`.


## License

[MIT License](LICENSE)
