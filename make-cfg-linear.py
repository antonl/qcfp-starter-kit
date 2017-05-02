from pyqcfp.runqcfp import QcfpConfig


cfg = QcfpConfig.from_yaml_file('template-cfg.yaml')
cfg.simulation_type = 'absorption'
cfg.simulation_code = 'excitons-abs'

with open('template-cfg.yaml', 'w') as f:
    cfg.to_yaml(stream=f)
