# 场景风险隔离版本（2026-10-19）

保留原去重版本的7507张接受图片；37组74张冲突版本继续隔离。491张变更分区，图片和标签字节不变。20对人工核查为10同组、9不同组、1不确定；20466对检索候选作为保守风险约束同分区放置，候选不等于确认同场景。全部1999张PartA教室来源图片放入训练集。

| 分区 | 图片 | 戴帽头部框 | 未戴帽头部框 |
| --- | ---: | ---: | ---: |
| train | 6488 | 7248 | 102502 |
| val | 511 | 762 | 4623 |
| test | 508 | 946 | 3929 |

已检索候选与已确认同组对的跨分区数均为0；检索并非穷尽。不同场景可因保守约束同处风险桶，不将人工“不同组”改成“同场景”。val/test不含PartA，不能声称验证了新教室泛化。

## 隔离前后的直接效果（2026-10-23补充核验）

这里的“前”是第一次内容去重后的7507张版本，不是原始历史训练划分。

| 指标 | 隔离前 | 隔离后 |
| --- | ---: | ---: |
| 已检索关联候选跨分区 | 20,466对 | 0对 |
| 人工确认同组的跨分区关系 | 10对 | 0对 |
| PartA训练／验证／测试图片 | 1593／206／200 | 1999／0／0 |
| 全部训练／验证／测试图片 | 6005／750／752 | 6488／511／508 |
| 接受图片总数 | 7,507 | 7,507 |

共491张改变归属；清单中的图片和标签哈希逐项一致。这是已有分组结果的重新核验，没有重新划分、改标注或训练。

20,466对由2,126张不同图片构成，同一图片可与多张图片配对。该数值是固定检索候选中的关系计数，不是20,466张图片或20,466次确认泄漏。人工“不同组”判断仍保留；保守风险桶可以包含不同场景。

用仓库已公开的旧清单、新清单、隔离约束和人工判断即可重算，无需原图或安装检测框架：

```bash
python reports/scene_isolation/recompute_effect.py
```

[重算脚本](recompute_effect.py)先校验四份证据的SHA256，再计算[前后指标JSON](before_after.json)。它核对清单记录的内容哈希，原图字节的既有检查见[构建验证](validation.json)。

这些结果衡量数据质量和评价条件，不能换算成准确率提升。它们补充了[旧、新标准版训练协议变化](../baseline_evolution.md)；新标准版与P2的性能对照另见[训练报告](../p2_training_comparison/README.md)。


## 可核查文件

- [审计](dataset_audit.json)、[分区变更](split_changes.csv)、[原人工判断](review_decisions.json)、[验证摘要](validation.json)。
- [完整划分清单（gzip）](split_manifest.csv.gz)：解压后 SHA256 `5781ecdffa173010f539189b15565220f5e2ac597b5f8ea72f5cf7b1e8bf63e4`。
- [隔离约束（gzip）](isolation_constraints.json.gz)：解压后 SHA256 `afcacf6d160443637813879f37f4b6d507a05c75c9941c2271ebe8c500766efb`。
- [训练框尺寸诊断](training_resolution_diagnostic.json)：仅标注几何统计，不是模型准确率。

压缩仅减小文本体积，解压后字节与本地正式清单相同。原图与标签不再分发。这里保存的是最终分区证据；原始候选检索中间文件与完整交互核查页留在本地，不能凭这些公开摘要复演候选检索过程。

## 恢复固定分区用于训练

先按[工作流](../../docs/WORKFLOW.md)获取同一SHWD版本、转换并生成 `helmet_dataset_clean`。下面在仓库根目录执行，只创建新的独立目录，不重新检索或改变分区。缺图、哈希不符时停止；不要自行跳过。输出存在时拒绝覆盖。

```python
import csv, gzip, json, shutil
from pathlib import Path
import yaml
from workflow import sha256, verify_dataset

source = Path('helmet_dataset_clean')
evidence = Path('reports/scene_isolation')
out = Path('datasets/helmet_scene_isolated_v1')
assert not out.exists(), '输出已存在，先核实，不要覆盖'
source_identity = verify_dataset(source / 'data.yaml')
audit = json.loads((evidence / 'dataset_audit.json').read_text())
assert source_identity['manifest_sha256'] == audit['source_dataset_manifest_sha256']
with (source / 'split_manifest.csv').open(newline='') as f:
    original = {Path(r['image']).name: r for r in csv.DictReader(f)}
out.mkdir(parents=True)
(out / 'BUILD_INCOMPLETE.json').write_text('{}')
for name in ('split_manifest.csv', 'isolation_constraints.json'):
    (out / name).write_bytes(gzip.decompress((evidence / (name + '.gz')).read_bytes()))
for name in ('dataset_audit.json', 'review_decisions.json'):
    shutil.copy2(evidence / name, out / name)
assert sha256(out / 'split_manifest.csv') == audit['manifest_sha256']
with (out / 'split_manifest.csv').open(newline='') as f:
    for row in csv.DictReader(f):
        old = original[Path(row['image']).name]
        for key in ('image', 'label'):
            src, dst = source / old[key], out / row[key]
            assert dst.resolve().is_relative_to(out.resolve())
            assert sha256(src) == row[key + '_sha256']
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
(out / 'data.yaml').write_text(yaml.safe_dump({
    'path': str(out.resolve()), 'train': 'images/train',
    'val': 'images/val', 'test': 'images/test',
    'names': {0: 'hat', 1: 'person'}}))
(out / 'BUILD_INCOMPLETE.json').unlink()
print(verify_dataset(out / 'data.yaml'))
```

恢复保留了原审计作为历史来源，其中 `copy_methods` 描述最初Mac构建，不代表你本次复制方式。训练时显式指定数据，避免克隆后缺少本地登记文件而选回首轮划分：

```bash
python train_helmet.py --variant standard --data datasets/helmet_scene_isolated_v1/data.yaml --dry-run
python train_helmet.py --variant p2 --data datasets/helmet_scene_isolated_v1/data.yaml --dry-run
```

以上两条只预检。正式训练和固定初始化权重准备见[P2说明](../../docs/P2.md)。已有训练不因这次发布而需要重跑。
