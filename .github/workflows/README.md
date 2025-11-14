# LibAMM Build and Publish Workflow

## 架构说明

LibAMM 是一个独立的 C++ 库，作为 SAGE 的 submodule。它有自己的构建、测试和发布流程。

### 仓库关系

```
intellistream/SAGE (主仓库)
└── packages/sage-libs/
    └── src/sage/libs/
        └── libamm/ (submodule -> intellistream/LibAMM)
```

### 发布流程

1. **LibAMM 仓库** (`intellistream/LibAMM`)
   - 维护源代码和 C++ 测试
   - 构建预编译的 Python wheels
   - 发布到 PyPI 作为 `isage-libamm`

2. **SAGE 主仓库** (`intellistream/SAGE`)
   - 将 `isage-libamm` 声明为可选依赖
   - 用户安装 `isage-libs[amm]` 时自动从 PyPI 获取

## Workflow 说明

### 触发条件

- **自动触发**: Push 到 `main` 或 `main-dev` 分支（源代码变更时）
- **PR 检查**: 每个 PR 都会运行构建和测试，但不发布
- **手动触发**: 可选择目标仓库（PyPI/TestPyPI）

### 工作流程

```
1. check-version
   ├─ 读取 setup.py 中的版本号
   ├─ 对比最新的 Git tag
   └─ 决定是否需要发布

2. cpp-build-test
   ├─ 配置 CMake
   ├─ 编译 C++ 库
   └─ 运行 6 个 C++ 测试套件
      ✅ 所有测试必须通过才继续

3. build-wheels (parallel)
   ├─ Python 3.9
   ├─ Python 3.10
   ├─ Python 3.11
   └─ Python 3.12
      每个版本：
      ├─ 构建 wheel
      └─ 测试安装和 import

4. publish-to-pypi
   ├─ 收集所有 wheels
   ├─ 发布到 PyPI/TestPyPI
   ├─ 创建 Git tag (vX.Y.Z)
   └─ 创建 GitHub Release

5. summary
   └─ 生成构建摘要
```

### C++ 测试套件

必须全部通过的测试：
- `cpp_test` - 基础功能测试
- `sketch_test` - Sketch 算法测试
- `crs_test` - 稀疏矩阵测试
- `weighted_cr_test` - 加权测试
- `block_partition_test` - 分块测试
- `streaming_test` - 流式处理测试

## 版本管理

### 手动更新版本

在 `setup.py` 中修改版本号：

```python
setup(
    name='PyAMM',  # 会被 workflow 自动改为 isage-libamm
    version='0.1.5',  # 👈 修改这里
    ...
)
```

### 发布规则

- **版本变化 + Push** → 自动发布
- **版本未变化** → 跳过发布（仅运行测试）
- **main 分支** → PyPI（生产环境）
- **main-dev 分支** → TestPyPI（测试环境）

### 示例：发布新版本

1. 修改代码
2. 更新 `setup.py` 中的版本号（如 `0.1.5` → `0.1.6`）
3. Commit 并 push 到 `main-dev`
4. Workflow 自动：
   - ✅ 运行 C++ 测试
   - ✅ 构建 4 个 Python wheels
   - ✅ 发布到 TestPyPI
5. 测试通过后，merge 到 `main`
6. 自动发布到生产 PyPI

## 在 SAGE 中使用

### 安装 LibAMM

```bash
# 安装 sage-libs with LibAMM
pip install isage-libs[amm]

# 或单独安装 LibAMM
pip install isage-libamm
```

### 导入使用

```python
from sage.libs.libamm.python import PyAMM

# LibAMM 功能...
```

## 配置要求

### GitHub Secrets (LibAMM 仓库)

需要在 **LibAMM 仓库**中配置：

1. **PYPI_API_TOKEN** - PyPI API token
2. **TEST_PYPI_API_TOKEN** - TestPyPI API token
3. **PAT_TOKEN** - Personal Access Token (用于创建 tags)

### Self-hosted Runner

- **标签**: `self-hosted`, `Linux`, `X64`
- **内存**: ≥16GB（推荐 32GB）
- **CPU**: 多核（推荐 8+ 核）

## 故障排除

### C++ 测试失败

检查测试日志，定位失败的测试用例。所有测试必须通过才能发布。

### Wheel 构建失败（OOM）

增加 runner 内存或降低并行度：
```yaml
CMAKE_BUILD_PARALLEL_LEVEL: '1'  # 改为 1
```

### PyPI 发布失败

1. 检查版本号是否已经存在（PyPI 不允许重复上传）
2. 验证 API token 权限
3. 首次发布需要先在 PyPI 创建项目

## 参考链接

- **LibAMM 源码**: https://github.com/intellistream/LibAMM
- **PyPI 包**: https://pypi.org/project/isage-libamm/
- **SAGE 文档**: https://intellistream.github.io/SAGE-Pub/
