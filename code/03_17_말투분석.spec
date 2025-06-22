# 03_17_말투분석.spec
from PyInstaller.utils.hooks import collect_all

torchvision_collect = collect_all('torchvision')

a = Analysis(
    ['03_17_말투분석.py'],
    pathex=['C:\\Users\\joheo\\OneDrive\\Desktop\\개인 코딩\\유튜브'],
    binaries=torchvision_collect[1],
    datas=torchvision_collect[0],
    hiddenimports=torchvision_collect[2] + ['torch', 'torchvision', 'torchvision.io'],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='03_17_말투분석',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)
