import json
import base64


def jsonStringify(data):
    return json.dumps(data)


def build(data):
    jsonData = json.dumps(data)
    base64Data = base64.b64encode(jsonData.encode('utf-8')).decode('utf-8')
    return base64Data


def result(config, data):
    print(f'Result[{config["id"]}][{jsonStringify(data)}]')
    print(f'AigcPanelRunResult[{config["id"]}][{build(data)}]')
