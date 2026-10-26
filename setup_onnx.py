"""Prepare fixed, optional CPU deployment packages in .onnx_deps only."""
import bootstrap
import json
import requests
from download_resources import download,extract_checked


def main():
    target=bootstrap.ROOT/'.onnx_deps';target.mkdir(exist_ok=True)
    manifest={}
    pins=[('onnx','1.17.0'),('onnxruntime','1.20.1'),('protobuf','5.29.3'),
          ('flatbuffers','25.2.10'),('coloredlogs','15.0.1'),('humanfriendly','10.0')]
    for name,version in pins:
        response=requests.get(f'https://pypi.org/pypi/{name}/{version}/json',timeout=30)
        response.raise_for_status()
        choices=[r for r in response.json()['urls'] if r['filename'].endswith(('.whl')) and
                 any(r['filename'].endswith(tag) for tag in ['py3-none-any.whl','py2.py3-none-any.whl',
                     'cp312-cp312-win_amd64.whl','cp310-abi3-win_amd64.whl'])]
        priority=lambda r: (2 if r['filename'].endswith('cp312-cp312-win_amd64.whl') else
                            1 if r['filename'].endswith('cp310-abi3-win_amd64.whl') else 0)
        choices.sort(key=priority,reverse=True)
        if not choices or (len(choices)>1 and priority(choices[0])==priority(choices[1])):
            raise RuntimeError(f'Expected a unique preferred Python3.12 Windows wheel: {name}')
        record=choices[0]
        wheel=bootstrap.ROOT/'.downloads'/record['filename']
        info=download(record['url'],wheel)
        assert info['sha256_recorded_locally']==record['digests']['sha256']
        extract_checked(wheel,target)
        manifest[name]={'version':version,'wheel':record['filename'],'url':record['url'],
            'pypi_sha256':record['digests']['sha256'],'verified':True}
        print('ONNX_DEPENDENCY',name,version,flush=True)
    (bootstrap.ROOT/'results/onnx_package_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


if __name__=='__main__':main()
