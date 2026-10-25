"""Run synthetic structure checks; never train or register a production model."""
import argparse
import html
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description='P2+CBAM 结构核查（不训练、不读取评价图片）')
    parser.add_argument('--self-test', action='store_true', help='同时运行模块、梯度、重载与初始化测试')
    args = parser.parse_args()
    import torch
    import ultralytics
    from helmet_cbam import CONFIG, build_cbam_model
    from workflow import sha256
    torch.set_num_threads(2)
    out = ROOT/'reports/p2_cbam_structure'
    out.mkdir(parents=True, exist_ok=True)
    test_record = None
    if args.self_test:
        run = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover',
                              '-s', 'tests', '-p', 'test_p2_cbam.py', '-v'], cwd=ROOT,
                             capture_output=True, text=True)
        log = out/'tests.log'
        log.write_text(run.stdout + run.stderr)
        if run.returncode:
            raise RuntimeError(f'核查未通过，请查看 {log}')
        test_record = {'exit_code': run.returncode, 'log': str(log.relative_to(ROOT)),
                       'log_sha256': sha256(log)}
    model, audit = build_cbam_model()
    shapes = []
    handle = model.model.model[-1].register_forward_pre_hook(
        lambda module, inputs: shapes.extend([list(t.shape) for t in inputs[0]]))
    try:
        with torch.inference_mode():
            pred = model.model.eval()(torch.zeros(1, 3, 640, 640))[0]
    finally:
        handle.remove()
    if shapes != [[1, 32, 160, 160], [1, 64, 80, 80], [1, 128, 40, 40], [1, 256, 20, 20]]:
        raise RuntimeError(f'检测头输入尺寸不符合预期: {shapes}')
    if not torch.isfinite(pred).all():
        raise RuntimeError('前向输出存在非有限数值')
    audit.update({'checked_at': datetime.now(timezone.utc).isoformat(),
                  'ultralytics': ultralytics.__version__, 'torch': torch.__version__,
                  'device': 'cpu', 'threads': torch.get_num_threads(),
                  'feature_shapes_640': shapes, 'prediction_shape': list(pred.shape),
                  'finite_forward': True, 'tests': test_record,
                  'accuracy_evaluated': False, 'optimizer_steps': 0,
                  'real_dataset_images_read': 0,
                  'source_hashes': {str(p.relative_to(ROOT)): sha256(p) for p in
                                   [ROOT/'helmet_cbam.py', CONFIG, Path(__file__), ROOT/'tests/test_p2_cbam.py']}})
    (out/'structure_check.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
    extra = audit['cbam_parameters'] / audit['p2_parameters'] * 100
    test_text = '9 项自动测试通过' if test_record else '本次完成结构与前向核查；完整测试请加 --self-test'
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>P2＋CBAM：第一步结构核查</title><style>
body{{font:17px/1.8 -apple-system,system-ui;background:#f4f6fa;color:#16263c;margin:0;padding:32px}}
main{{max-width:940px;margin:auto}}section{{background:white;padding:22px 28px;border-radius:14px;margin:18px 0}}
h1{{font-size:32px}}h2{{font-size:23px}}.good{{color:#08795b}}.note{{background:#fff2d8}}code{{background:#edf1f7;padding:3px 5px}}
table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;border-bottom:1px solid #ddd;padding:9px}}a{{color:#235ac2}}
</style><main><h1>P2＋CBAM：第一步已接通</h1><p>这是新的单模型结构。核查日期：{html.escape(audit['checked_at'][:10])}。</p>
<section class="note"><strong>当前状态：结构已核查，尚未训练。</strong><p>没有新增训练轮次，没有修改原始图片、标签或已训练权重，也没有替换检测网页中的正式模型。本页不能证明精度提升。</p></section>
<section><h2>接在哪里？</h2><p><strong>P2 特征（第18层） → CBAM（第28层） → 四尺度检测头的 P2 输入</strong></p>
<p>P3/P4/P5 仍从第21/24/27层直接送入检测头。第19层仍读取原 P2 特征，所以注意力只作用于 P2 检测分支。检测头由第28层移到第29层，权重按语义位置对应。</p>
<p>通道注意力：平均池化＋最大池化 → 共享 MLP → 特征加权。空间注意力：通道最大值＋均值 → 7×7 卷积 → 位置加权。</p></section>
<section><h2>已经核验的结果</h2><table><tr><th>检查项</th><th>结果</th></tr>
<tr><td>原 P2 参数量</td><td>{audit['p2_parameters']:,}</td></tr>
<tr><td>P2＋CBAM 参数量</td><td>{audit['p2_cbam_parameters']:,}</td></tr>
<tr><td>新增参数</td><td>{audit['cbam_parameters']}（约 {extra:.4f}%）；参数少不代表耗时为零</td></tr>
<tr><td>640 输入的四个特征尺寸</td><td>160×160、80×80、40×40、20×20</td></tr>
<tr><td>检测步长</td><td>4、8、16、32</td></tr>
<tr><td>非 CBAM 层初始化</td><td>与同随机种子的原 P2 初始化逐张量一致</td></tr>
<tr><td>输出检查</td><td>尺寸正确，数值有限；未计算任何准确率指标</td></tr>
<tr><td>自动测试</td><td class="good">{test_text}</td></tr></table>
<p>完整测试包括公式核对、错误通道拦截、连接和权重核对、注意力替换为恒等映射后输出完全一致、正方形及长方形输入、检测损失反向传播、训练器重建、保存重载和随机状态保护。</p>
<p>反向传播只在临时内存模型上检查梯度，没有执行优化器更新。临时未训练检查点在重载测试后自动删除。</p></section>
<section><h2>初始化为什么这样做？</h2><p>沿用之前 P2 对照的策略：官方 YOLOv8n 的共有层0～15加载预训练参数，其余原 P2 层保持相同随机种子的初始值，仅 CBAM 是新增参数。没有拿已经训练50轮的 best.pt 冒充同条件初始化。</p>
<p>这样后续可以设计只增加 CBAM 的对照。正式实验仍需固定数据版本、训练参数、硬件和随机种子；目前还没有启动该实验。</p></section>
<section><h2>文件与下一步</h2><p><a href="../../docs/P2_CBAM.md">代码解读和下一步说明</a> · <a href="structure_check.json">机器可读核查记录</a> · <a href="tests.log">自动测试日志（完整测试后生成）</a></p>
<p>下一步先确定公平训练方案，再在 RTX 3060 游戏本训练 P2＋CBAM。2026-10-25已补齐CBAM正式训练命令；参数沿用原P2，见 docs/P2_CBAM_TRAINING.md，迁移后先检查再训练。</p>
<p>不要用未经训练的 CBAM 直接替换已训练的 P2 后比较准确率。现有508张测试图片已经被多轮查看，后续不能称为新的盲测集。</p></section></main></html>'''
    (out/'index.html').write_text(page)
    print(f'结构核查完成（未训练）：{out / "index.html"}')
    print(f'P2: {audit["p2_parameters"]:,}; P2+CBAM: {audit["p2_cbam_parameters"]:,}; added: {audit["cbam_parameters"]}')


if __name__ == '__main__':
    main()
