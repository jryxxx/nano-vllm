# GitHub 发布记录

2026-11-15。发布到 `jryxxx/nano-vllm`。

- 推理模块 `nanovllm/**/*.py` 与最终实测归档逐字节一致；源码内容 SHA256（按路径排序拼接）为 `d0f36c37df987bec781942dab402776132f115eced8ac44e1bfb4c00a0be19b5`。
- 发布新增 `scripts/reproduce.sh`、安装依赖说明、数据说明及模型路径环境变量 `NANO_MODEL`。`run_final_suite.py` 使用当前 Python，并支持 `NANO_UPSTREAM` / `NANO_FLASHINFER_PATH`。
- 历史缓存回归对照改为仓库内 `tests/fixtures/pre_cache_policy_block_manager.py`，来自原实验服务器 `nano-vllm-pre-cache-policy` 快照；fixture 是测试对照，非运行时引擎。
- 发布入口在原远程 GPU 上执行缓存 logits 回归：oracle、previous_fifo、FIFO、LRU、leaf-LRU、cost 共 576 次检查通过。推理内核没有改动；性能沿用已验证的最终冻结实验，不将发布入口检查当作新的完整性能实验。
- 全部 Python 文件通过 AST 语法检查，Shell 入口语法通过，发布内容完成凭据模式扫描。原始数据字节保留，结果目录的 patch 空白/CSV CRLF 不做格式化。
- 原实验补丁、环境日志与路径保持原貌供追溯；它们还原实测快照，不包含发布入口适配。原始归档附在对应 GitHub Release 中，数据校验见 `results/PUBLIC_DATA_SHA256.json`。
