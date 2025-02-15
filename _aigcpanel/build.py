import json
import shutil

from .base import util

configFile = util.rootDir('_aigcpanel/config.json')
config = json.load(open(configFile, 'r', encoding='utf-8'))

config['platformName'] = util.platformName()
config['platformArch'] = util.platformArch()

if util.platformName() == 'win':
    config['launcher']['entry'] = f".\\_aienv\\python.exe"

archPath = f"{config['platformName']}-{config['platformArch']}"

# copy binary/osx-arm64/* to binary/
# util.copyAll(util.rootDir(f'_aigcpanel/binary/{archPath}'), util.rootDir('binary'))

# move binary launcher to root
# shutil.move(util.binaryPath('binary/launcher'), util.binaryPath('launcher'))

outputFile = util.rootDir('config.json')
json.dump(config, open(outputFile, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)

print(f"{config['platformName']}-{config['platformArch']}-v{config['version']}")
