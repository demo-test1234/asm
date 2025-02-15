from modelscope import snapshot_download

model_dir = snapshot_download(
    'MZ0x01/aigcpanel-server-musetalk',
    local_dir='aigcpanelmodels'
)
