# 在 AutoDL (RTX 3090 24GB) 环境复现具身智能目标导航项目 VLFM (Visual-Language Frontier Map) 的典型问题。

## C++ 扩展编译与 Python 路径遮蔽冲突

* **问题现象**：运行评测时报错：
  `ImportError: cannot import name 'filter_depth' from 'depth_camera_filtering' (unknown location)`
* **根本原因**：
  1. **缺少 C++ `.so` 库**：`depth_camera_filtering` 包含 C++ / Pybind11 底层代码，必须编译生成 `.so` 动态库。
  <!-- 2. **Python 名称遮蔽（Shadowing）**：主目录 `/root/autodl-tmp/vlfm` 下存在同名源码文件夹 `depth_camera_filtering`，Python 启动时优先读取该未编译的空壳文件夹，忽略了 Conda 环境。 -->
  3. **可编辑安装（`pip install -e`）限制**：软链接依赖于本地文件夹，导致直接重命名本地文件夹会报 `ModuleNotFoundError`。
* **解决方案**：
  1. 进入子目录全量独立安装：`cd depth_camera_filtering && pip install .`，将编译生成的 `.so` 动态库直接写入 Conda 的 `site-packages`。
  <!-- 2. 隔离本地源码文件夹：切回主目录执行 `mv depth_camera_filtering _depth_camera_filtering_bak`，消除路径遮蔽。 -->


## Hydra 框架缺失自定义组件的 YAML 配置文件

* **问题现象**：评测启动时连续报错：
  * `Could not load 'habitat/task/measurements/frontier_exploration_map'`
  * `Could not load 'habitat/task/lab_sensors/frontier_sensor'`
  * `Could not load 'habitat/task/lab_sensors/heading_sensor'`
* **根本原因**：VLFM 扩展了自定义的 Sensor 和 Task Measurement，并在实验配置 `vlfm_objectnav_hm3d.yaml` 中引用了它们。但项目模板中的 `config/habitat/task/` 对应子目录为空，缺少对应的 YAML 声明文件。
* **解决方案**：
  在主配置目录下建立对应的配置文件，指定组件类型：
  * 创建 `/root/autodl-tmp/vlfm/config/habitat/task/measurements/frontier_exploration_map.yaml`（`type: FrontierExplorationMap`）
  * 创建 `/root/autodl-tmp/vlfm/config/habitat/task/lab_sensors/frontier_sensor.yaml`（`type: FrontierSensor`）
  * 创建 `/root/autodl-tmp/vlfm/config/habitat/task/lab_sensors/heading_sensor.yaml`（`type: HeadingSensor`）


## 多进程 Worker 子进程组件未注册问题

* **问题现象**：仿真任务初始化时，多进程抛出断言崩溃：
  `AssertionError: invalid frontier_exploration_map type FrontierExplorationMap`
* **根本原因**：Habitat 在运行评测时开启了多进程向量化环境（`ForkServerProcess`）。子进程启动时是全新的 Python 解释器上下文，如果在子进程中**没有提前 `import` 加载** `frontier_exploration` 的代码，装饰器 `@registry.register_measure` 和 `@registry.register_sensor` 就不会在子进程中触发，导致子进程内存注册表中缺失这些类。
* **解决方案**：
  在 `vlfm/__init__.py` 中追加顶层显式导入，确保多进程 Worker 在导入 `vlfm` 时自动触发注册：
  ```python
  try:
      import frontier_exploration.measurements
      import frontier_exploration.frontier_sensor
  except Exception as e:
      pass
  ```