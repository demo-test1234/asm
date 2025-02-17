import sys, os, datetime, random, platform, shutil


def banner(param: dict):
    print('##################### MSBrick Info #####################')
    print('root:', root())
    if 'args' in param:
        print('args:', param['args'])
    print('######################################################')


def root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def rootDir(path):
    return root() + os.sep + path


def binaryPath(path):
    p = rootDir(path)
    if sys.platform == 'win32':
        p += '.exe'
    return p


def datetimeRandomName():
    return datetime.datetime.now().strftime('%Y%m%d%H%M%S_') + str(random.randint(1000, 9999))


def randomString(length=32):
    return ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=length))


def datetimeRandomNameParseTimestamp(name):
    if len(name) < 14:
        return 0
    return datetime.datetime.strptime(name[:14], '%Y%m%d%H%M%S').timestamp()


def platformName():
    os_name = sys.platform
    if os_name == 'darwin':
        os_name = 'osx'
    elif os_name == 'win32':
        os_name = 'win'
    elif os_name == 'linux':
        os_name = 'linux'
    else:
        os_name = 'unknown'
    return os_name


def platformArch():
    arch = platform.machine().lower()
    if arch in ['x86_64', 'amd64']:
        arch = 'x86'
    elif arch in ['arm64', 'aarch64']:
        arch = 'arm64'
    else:
        arch = 'unknown'
    return arch


def copyAll(src, dst):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        src_path = os.path.join(src, item)
        dest_path = os.path.join(dst, item)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dest_path)
        else:
            copyAll(src_path, dest_path)
