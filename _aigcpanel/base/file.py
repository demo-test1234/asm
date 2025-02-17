import hashlib
import json
import os
import time
import requests

import _aigcpanel.base.util


def getCacheRoot():
    cacheRoot = _aigcpanel.base.util.rootDir('_cache/file')
    if not os.path.exists(cacheRoot):
        os.makedirs(cacheRoot)
    return cacheRoot


def contentFromUrl(url):
    headers = {
        'User-Agent': 'AigcPanelServer'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.text
    else:
        print('contentFromUrl.error', response.status_code, response.text)
        return None


def contentText(pathOrUrl):
    if pathOrUrl.startswith('http'):
        return contentFromUrl(pathOrUrl)
    else:
        with open(pathOrUrl, 'r', encoding='utf-8') as file:
            return file.read()
    raise ValueError(f"Invalid path or URL: {pathOrUrl}")


def contentJson(pathOrUrl):
    content = contentText(pathOrUrl)
    if content:
        return json.loads(content)
    else:
        return None


def downloadFileDirect(url, path):
    cleanCache()
    headers = {
        'User-Agent': 'AigcPanelServer'
    }
    response = requests.get(url, headers=headers, stream=True)
    if response.status_code == 200:
        with open(path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        return
    raise ValueError(f"Failed to download {url}. HTTP status code: {response.status_code}")


def cleanCache():
    cacheRoot = getCacheRoot()
    for filename in os.listdir(cacheRoot):
        path = os.path.join(cacheRoot, filename)
        if os.path.isfile(path):
            if time.time() - os.path.getatime(path) > 3600 * 24 * 7:
                os.remove(path)


def localCache(pathOrUrl):
    if pathOrUrl.startswith('http'):
        cacheRoot = getCacheRoot()
        md5 = hashlib.md5(pathOrUrl.encode('utf-8')).hexdigest()
        ext = pathOrUrl.split('.')[-1]
        cachePath = os.path.join(cacheRoot, f'{md5}.{ext}')
        if not os.path.exists(cachePath):
            downloadFileDirect(pathOrUrl, cachePath)
        else:
            os.utime(cachePath, None)
        if not os.path.exists(cachePath):
            raise ValueError(f"Failed to download {pathOrUrl}")
        return cachePath
    else:
        return pathOrUrl


def localCacheRandomPath(ext):
    cacheRoot = getCacheRoot()
    md5 = hashlib.md5(str(time.time()).encode('utf-8')).hexdigest()
    return os.path.join(cacheRoot, f'{md5}.{ext}')


def upload(config, localFile, remotePath):
    try:
        if config['type'] == 'AliyunOss':
            import oss2
            auth = oss2.Auth(config['accessKeyId'], config['accessKeySecret'])
            bucket = oss2.Bucket(auth, config['endpoint'], config['bucket'])
            bucket.put_object_from_file(remotePath, localFile)
            return f"{config['url']}/{remotePath}"
        elif config['type'] == 'QiniuKodo':
            from qiniu import Auth, put_file
            q = Auth(config['ak'], config['sk'])
            token = q.upload_token(config['bucket'], remotePath, 3600)
            ret, info = put_file(token, remotePath, localFile, version='v2')
            return f"{config['url']}/{remotePath}"
        elif config['type'] == 'QcloudCos':
            from qcloud_cos import CosConfig
            from qcloud_cos import CosS3Client
            qconfig = CosConfig(Region=config['region'],
                                SecretId=config['secretId'],
                                SecretKey=config['secretKey'],
                                Token=None)
            client = CosS3Client(qconfig)
            with open(localFile, 'rb') as fp:
                response = client.put_object(
                    Bucket=config['bucket'],
                    Body=fp,
                    Key=remotePath,
                    StorageClass='STANDARD',
                    EnableMD5=False
                )
            print('response', response)
            return f"{config['url']}/{remotePath}"
        else:
            raise ValueError(f"Unsupported upload type: {config['type']}")
    except Exception as e:
        print(f"Error uploading file to: {e}")


def uploadToRandom(config, localFile, delete=True):
    ext = localFile.split('.')[-1]
    ossPath = 'temp/' + _aigcpanel.base.util.randomString() + '.' + ext
    result = upload(config, localFile, ossPath)
    if delete:
        os.remove(localFile)
    return result


def urlForResult(config, url):
    if config['mode'] == 'schedule':
        url = _aigcpanel.base.file.uploadToRandom(config['uploadConfig'], url)
    return url
