from datetime import datetime


def info(msg, *args):
    now = datetime.now()
    print(now, '-', msg, *args)


if __name__ == '__main__':
    info('aaa')
    info('aaa', {'bbb': 'ccc'})
    info('aaa', 'ccc', {'bbb': 'ccc'})
