#!/usr/bin/env python
import argparse
import sys
import os
import importlib.util

# 添加项目路径到 sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 加载完整的 torchlight 包（使用本地版本）
torchlight_dir = os.path.join(project_root, 'torchlight', 'torchlight')
torchlight_parent = os.path.join(project_root, 'torchlight')

# torchlight 目录放在最前面，优先于系统安装的版本
if torchlight_parent in sys.path:
    sys.path.remove(torchlight_parent)
sys.path.insert(0, torchlight_parent)

# 如果 torchlight 已经在 sys.modules 中，删除它以便重新加载
if 'torchlight' in sys.modules:
    # 删除 torchlight 及其子模块
    keys_to_remove = [k for k in sys.modules.keys() if k.startswith('torchlight')]
    for k in keys_to_remove:
        del sys.modules[k]

# 现在正常导入 torchlight（将使用本地版本）
import torchlight
from torchlight import import_class

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Processor collection')

    # region register processor yapf: disable
    processors = dict()
    processors['recognition'] = import_class('processor.recognition.REC_Processor')
    # 可选processor，如果导入失败则跳过
    try:
        processors['demo_old'] = import_class('processor.demo_old.Demo')
    except ImportError:
        pass
    try:
        processors['demo'] = import_class('processor.demo_realtime.DemoRealtime')
    except ImportError:
        pass
    try:
        processors['demo_offline'] = import_class('processor.demo_offline.DemoOffline')
    except ImportError:
        pass
    try:
        processors['demo_json'] = import_class('processor.demo_json.DemoJson')
    except ImportError:
        pass
    try:
        processors['demo_alpha_offline'] = import_class('processor.demo_alpha_offline.DemoAlphaOffline')
    except ImportError:
        pass
    #endregion yapf: enable

    # add sub-parser
    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])

    # read arguments
    arg = parser.parse_args()

    # start
    Processor = processors[arg.processor]
    p = Processor(sys.argv[2:])

    p.start()
