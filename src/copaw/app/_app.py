# -*- coding: utf-8 -*-
# pylint: disable=redefined-outer-name,unused-argument
"""
CoPaw 主 FastAPI 应用。

该模块主要负责把以下部分串联起来：
- 代理运行时（`AgentRunner` + `AgentApp.router`）
- 可热重载的管理器（配置监听、定时任务、渠道连接）
- 可选的 MCP 客户端支持
- Web 控制台静态 SPA（来自已打包资源或 `COPAW_CONSOLE_STATIC_DIR`）
"""
import asyncio
import mimetypes
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from agentscope_runtime.engine.app import AgentApp

from .runner import AgentRunner
from ..config import (  # pylint: disable=no-name-in-module
    load_config,
    update_last_dispatch,
    ConfigWatcher,
)
from ..config.utils import get_jobs_path, get_chats_path, get_config_path
from ..constant import DOCS_ENABLED, LOG_LEVEL_ENV, CORS_ORIGINS, WORKING_DIR
from ..__version__ import __version__
from ..utils.logging import setup_logger, add_copaw_file_handler
from .channels import ChannelManager  # pylint: disable=no-name-in-module
from .channels.utils import make_process_from_runner
from .mcp import MCPClientManager, MCPConfigWatcher  # MCP hot-reload support
from .runner.repo.json_repo import JsonChatRepository
from .crons.repo.json_repo import JsonJobRepository
from .crons.manager import CronManager
from .runner.manager import ChatManager
from .routers import router as api_router
from .routers.voice import voice_router
from ..envs import load_envs_into_environ
from ..providers.provider_manager import ProviderManager

# 在模块加载时应用日志等级，确保热重载/子进程继承与 CLI 启动一致的日志级别。
logger = setup_logger(os.environ.get(LOG_LEVEL_ENV, "info"))

# 确保静态资源对浏览器的 MIME 类型在不同平台上一致可用
# （例如某些环境下 Windows 可能缺少 `.js/.mjs` 的映射）。
mimetypes.init()
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("application/wasm", ".wasm")

# 在模块导入阶段把已持久化的环境变量加载到 `os.environ`，
# 以便 `lifespan` 启动流程开始时即可直接读取到这些值。
load_envs_into_environ()

# 核心运行时对象：
# - `runner` 负责代理执行/会话，并托管其委派的各类管理器。
# - `agent_app` 暴露代理相关的 API 路由供 `app.include_router` 挂载。
runner = AgentRunner()

agent_app = AgentApp(
    app_name="Friday",
    app_description="A helpful assistant",
    runner=runner,
)


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):  # pylint: disable=too-many-statements,too-many-branches
    """FastAPI 生命周期（lifespan）。

    启动阶段：按依赖顺序启动各个运行时管理器，并把它们挂载到
    `app.state` 上，供后续 API/任务使用。

    退出阶段：按相反顺序停止管理器，最后停止底层 `runner`。
    """
    startup_start_time = time.time()
    add_copaw_file_handler(WORKING_DIR / "copaw.log")
    await runner.start()

    # --- MCP 客户端管理器初始化（独立模块，可热重载） ---
    config = load_config()
    mcp_manager = MCPClientManager()
    if hasattr(config, "mcp"):
        try:
            await mcp_manager.init_from_config(config.mcp)
            logger.debug("MCP client manager initialized")
        except BaseException as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            logger.exception("Failed to initialize MCP manager")
    runner.set_mcp_manager(mcp_manager)

    # --- 渠道连接器初始化/启动（依据 config.json 配置） ---
    channel_manager = ChannelManager.from_config(
        process=make_process_from_runner(runner),
        config=config,
        on_last_dispatch=update_last_dispatch,
    )
    await channel_manager.start_all()

    # --- 定时任务（cron）初始化/启动 ---
    repo = JsonJobRepository(get_jobs_path())
    cron_manager = CronManager(
        repo=repo,
        runner=runner,
        channel_manager=channel_manager,
        timezone="UTC",
    )
    await cron_manager.start()

    # --- 聊天管理器初始化，并连接到 runner 的会话（session） ---
    chat_repo = JsonChatRepository(get_chats_path())
    chat_manager = ChatManager(
        repo=chat_repo,
    )

    runner.set_chat_manager(chat_manager)

    # --- 配置文件监听器（配置变更后热重载 channels，并刷新相关心跳/状态） ---
    config_watcher = ConfigWatcher(
        channel_manager=channel_manager,
        cron_manager=cron_manager,
    )
    await config_watcher.start()

    # --- MCP 配置监听器（MCP 配置变更时自动重载 MCP 客户端） ---
    mcp_watcher = None
    if hasattr(config, "mcp"):
        try:
            mcp_watcher = MCPConfigWatcher(
                mcp_manager=mcp_manager,
                config_loader=load_config,
                config_path=get_config_path(),
            )
            await mcp_watcher.start()
            logger.debug("MCP config watcher started")
        except BaseException as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            logger.exception("Failed to start MCP watcher")

    # 将 channel_manager 注入审批服务：
    # 使审批服务可以主动把审批消息推送到诸如钉钉等外部渠道。
    from .approvals import get_approval_service

    get_approval_service().set_channel_manager(channel_manager)

    # --- 模型提供者管理器（不参与热重载，常驻内存） ---
    provider_manager = ProviderManager.get_instance()

    # 把关键管理器对象暴露给各个 endpoint/任务使用：
    # 统一放入 app.state 作为进程内共享引用。
    app.state.runner = runner
    app.state.channel_manager = channel_manager
    app.state.cron_manager = cron_manager
    app.state.chat_manager = chat_manager
    app.state.config_watcher = config_watcher
    app.state.mcp_manager = mcp_manager
    app.state.mcp_watcher = mcp_watcher
    app.state.provider_manager = provider_manager

    _restart_task: asyncio.Task | None = None

    async def _restart_services() -> None:
        """停止所有管理器，然后从配置重建（不退出进程）。

        设计要点：
        - Single-flight：同一时间只允许一个重启流程执行；并发触发或重复触发方会等待
          当前重启完成，然后复用同一轮重启结果，避免“重复重启/重复释放资源”。
        - asyncio.shield()：当调用方协程被取消（例如某个渠道请求因超时/取消而取消），
          重启任务仍继续运行，避免取消信号级联到更深层协程树，从而降低取消导致的异常风险
          （包括可能出现的 RecursionError）。
        """
        # pylint: disable=too-many-statements
        nonlocal _restart_task
        # 调用方任务（位于 agent_app 的 _local_tasks 中）需要保持可运行，
        # 从而确保其能够输出/返回最终的“重启完成”消息给调用链路。
        restart_requester_task = asyncio.current_task()

        async def _run_then_clear() -> None:
            try:
                await _do_restart_services(
                    restart_requester_task=restart_requester_task,
                )
            finally:
                nonlocal _restart_task
                _restart_task = None

        if _restart_task is not None and not _restart_task.done():
            logger.info(
                "_restart_services: waiting for in-progress restart to finish",
            )
            await asyncio.shield(_restart_task)
            return
        if _restart_task is not None and _restart_task.done():
            _restart_task = None
        logger.info("_restart_services: starting restart")
        _restart_task = asyncio.create_task(_run_then_clear())
        await asyncio.shield(_restart_task)

    async def _teardown_new_stack(
        mcp_watcher=None,
        config_watcher=None,
        cron_mgr=None,
        ch_mgr=None,
        mcp_mgr=None,
    ) -> None:
        """新栈启动回滚：按与启动相反的顺序停止已启动的组件。"""
        if mcp_watcher is not None:
            try:
                await mcp_watcher.stop()
            except Exception:
                logger.debug(
                    "rollback: mcp_watcher.stop failed",
                    exc_info=True,
                )
        if config_watcher is not None:
            try:
                await config_watcher.stop()
            except Exception:
                logger.debug(
                    "rollback: config_watcher.stop failed",
                    exc_info=True,
                )
        if cron_mgr is not None:
            try:
                await cron_mgr.stop()
            except Exception:
                logger.debug(
                    "rollback: cron_manager.stop failed",
                    exc_info=True,
                )
        if ch_mgr is not None:
            try:
                await ch_mgr.stop_all()
            except Exception:
                logger.debug(
                    "rollback: channel_manager.stop_all failed",
                    exc_info=True,
                )
        if mcp_mgr is not None:
            try:
                await mcp_mgr.close_all()
            except Exception:
                logger.debug(
                    "rollback: mcp_manager.close_all failed",
                    exc_info=True,
                )

    async def _do_restart_services(
        restart_requester_task: asyncio.Task | None = None,
    ) -> None:
        """重启流程核心：先取消正在执行的代理请求，再切换到新管理器栈。

        目的：
        - 让“正在飞的”请求尽快停止（从而能把失败/错误状态回传到渠道）；
        - 释放旧栈资源并停止旧的管理器实例；
        - 基于最新配置启动新栈，随后把 app.state/runner 的引用切换到新实例。
        """
        # pylint: disable=too-many-statements
        try:
            config = load_config(get_config_path())
        except Exception:
            logger.exception("restart_services: load_config failed")
            return

        # 1) 取消在飞的代理请求。
        #    不等待它们“完全退出”，避免控制台触发的重启任务被阻塞。
        #    同时也减少某些任务在取消后退出较慢时带来的“等待-取消”死锁风险。
        local_tasks = getattr(agent_app, "_local_tasks", None)
        if local_tasks:
            to_cancel = [
                t
                for t in list(local_tasks.values())
                if t is not restart_requester_task and not t.done()
            ]
            for t in to_cancel:
                t.cancel()
            if to_cancel:
                logger.info(
                    "restart: cancelled %s in-flight task(s), not waiting",
                    len(to_cancel),
                )

        # 2) 停止旧栈：按约定的停止顺序释放旧的管理器实例。
        cfg_w = app.state.config_watcher
        mcp_w = getattr(app.state, "mcp_watcher", None)
        cron_mgr = app.state.cron_manager
        ch_mgr = app.state.channel_manager
        mcp_mgr = app.state.mcp_manager
        try:
            await cfg_w.stop()
        except Exception:
            logger.exception(
                "restart_services: old config_watcher.stop failed",
            )
        if mcp_w is not None:
            try:
                await mcp_w.stop()
            except Exception:
                logger.exception(
                    "restart_services: old mcp_watcher.stop failed",
                )
        try:
            await cron_mgr.stop()
        except Exception:
            logger.exception(
                "restart_services: old cron_manager.stop failed",
            )
        try:
            await ch_mgr.stop_all()
        except Exception:
            logger.exception(
                "restart_services: old channel_manager.stop_all failed",
            )
        if mcp_mgr is not None:
            try:
                await mcp_mgr.close_all()
            except Exception:
                logger.exception(
                    "restart_services: old mcp_manager.close_all failed",
                )

        # 3) 基于最新配置构建并启动新栈，然后在成功后替换引用。
        new_mcp_manager = MCPClientManager()
        if hasattr(config, "mcp"):
            try:
                await new_mcp_manager.init_from_config(config.mcp)
            except Exception:
                logger.exception(
                    "restart_services: mcp init_from_config failed",
                )
                return

        new_channel_manager = ChannelManager.from_config(
            process=make_process_from_runner(runner),
            config=config,
            on_last_dispatch=update_last_dispatch,
        )
        try:
            await new_channel_manager.start_all()
        except Exception:
            logger.exception(
                "restart_services: channel_manager.start_all failed",
            )
            await _teardown_new_stack(mcp_mgr=new_mcp_manager)
            return

        job_repo = JsonJobRepository(get_jobs_path())
        new_cron_manager = CronManager(
            repo=job_repo,
            runner=runner,
            channel_manager=new_channel_manager,
            timezone="UTC",
        )
        try:
            await new_cron_manager.start()
        except Exception:
            logger.exception(
                "restart_services: cron_manager.start failed",
            )
            await _teardown_new_stack(
                ch_mgr=new_channel_manager,
                mcp_mgr=new_mcp_manager,
            )
            return

        new_config_watcher = ConfigWatcher(
            channel_manager=new_channel_manager,
            cron_manager=new_cron_manager,
        )
        try:
            await new_config_watcher.start()
        except Exception:
            logger.exception(
                "restart_services: config_watcher.start failed",
            )
            await _teardown_new_stack(
                cron_mgr=new_cron_manager,
                ch_mgr=new_channel_manager,
                mcp_mgr=new_mcp_manager,
            )
            return

        new_mcp_watcher = None
        if hasattr(config, "mcp"):
            try:
                new_mcp_watcher = MCPConfigWatcher(
                    mcp_manager=new_mcp_manager,
                    config_loader=load_config,
                    config_path=get_config_path(),
                )
                await new_mcp_watcher.start()
            except Exception:
                logger.exception(
                    "restart_services: mcp_watcher.start failed",
                )
                await _teardown_new_stack(
                    config_watcher=new_config_watcher,
                    cron_mgr=new_cron_manager,
                    ch_mgr=new_channel_manager,
                    mcp_mgr=new_mcp_manager,
                )
                return

        if hasattr(config, "mcp"):
            runner.set_mcp_manager(new_mcp_manager)
            app.state.mcp_manager = new_mcp_manager
            app.state.mcp_watcher = new_mcp_watcher
        else:
            runner.set_mcp_manager(None)
            app.state.mcp_manager = None
            app.state.mcp_watcher = None
        app.state.channel_manager = new_channel_manager
        app.state.cron_manager = new_cron_manager
        app.state.config_watcher = new_config_watcher
        logger.info("Daemon restart (in-process) completed: managers rebuilt")

    setattr(runner, "_restart_callback", _restart_services)

    startup_elapsed = time.time() - startup_start_time
    logger.debug(
        f"Application startup completed in {startup_elapsed:.3f} seconds",
    )

    try:
        yield
    finally:
        # 停止当前 `app.state` 持有的管理器引用：
        # 如果发生过进程内重启，这里可能引用的是“重启后的实例”。
        cfg_w = getattr(app.state, "config_watcher", None)
        mcp_w = getattr(app.state, "mcp_watcher", None)
        cron_mgr = getattr(app.state, "cron_manager", None)
        ch_mgr = getattr(app.state, "channel_manager", None)
        mcp_mgr = getattr(app.state, "mcp_manager", None)
        # 停止顺序：监听器 -> 定时任务 -> 渠道 -> MCP -> runner
        if cfg_w is not None:
            try:
                await cfg_w.stop()
            except Exception:
                pass
        if mcp_w is not None:
            try:
                await mcp_w.stop()
            except Exception:
                pass
        if cron_mgr is not None:
            try:
                await cron_mgr.stop()
            except Exception:
                pass
        if ch_mgr is not None:
            try:
                await ch_mgr.stop_all()
            except Exception:
                pass
        if mcp_mgr is not None:
            try:
                await mcp_mgr.close_all()
            except Exception:
                pass
        await runner.stop()


app = FastAPI(
    lifespan=lifespan,
    docs_url="/docs" if DOCS_ENABLED else None,
    redoc_url="/redoc" if DOCS_ENABLED else None,
    openapi_url="/openapi.json" if DOCS_ENABLED else None,
)

# 如果配置了 `CORS_ORIGINS`，则启用 CORS 中间件以支持跨域访问。
if CORS_ORIGINS:
    origins = [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# 控制台静态目录解析策略：优先 `env`，其次是打包后的 `copaw` 资源（`console`），最后回退到 `cwd`。
_CONSOLE_STATIC_ENV = "COPAW_CONSOLE_STATIC_DIR"


def _resolve_console_static_dir() -> str:
    """解析 Web 控制台静态资源目录。

    优先级顺序：
    1. `COPAW_CONSOLE_STATIC_DIR` 环境变量（显式覆盖）
    2. 已打包发行版中的 `copaw/console` 目录
    3. 本地工作目录的兜底路径（开发/构建产物）
    """
    if os.environ.get(_CONSOLE_STATIC_ENV):
        return os.environ[_CONSOLE_STATIC_ENV]
    # 已发布发行版中，console 静态资源以“包内静态数据”的形式存在（不是 Python 包模块）。
    pkg_dir = Path(__file__).resolve().parent.parent
    candidate = pkg_dir / "console"
    if candidate.is_dir() and (candidate / "index.html").exists():
        return str(candidate)
    # 兼容/兜底逻辑：下一次发布后预计可移除
    # （因为届时 `vite` 会直接把 console 输出到 `src/copaw/console/` 目录）。
    cwd = Path(os.getcwd())
    for subdir in ("console/dist", "console_dist"):
        candidate = cwd / subdir
        if candidate.is_dir() and (candidate / "index.html").exists():
            return str(candidate)
    return str(cwd / "console" / "dist")


_CONSOLE_STATIC_DIR = _resolve_console_static_dir()
_CONSOLE_INDEX = (
    Path(_CONSOLE_STATIC_DIR) / "index.html" if _CONSOLE_STATIC_DIR else None
)
logger.info(f"STATIC_DIR: {_CONSOLE_STATIC_DIR}")


@app.get("/")
def read_root():
    """提供 Web 控制台首页；若资源不存在则返回提示信息。"""
    if _CONSOLE_INDEX and _CONSOLE_INDEX.exists():
        return FileResponse(_CONSOLE_INDEX)
    return {
        "message": (
            "CoPaw Web Console is not available. "
            "If you installed CoPaw from source code, please run "
            "`npm ci && npm run build` in CoPaw's `console/` "
            "directory, and restart CoPaw to enable the web console."
        ),
    }


@app.get("/api/version")
def get_version():
    """返回当前 CoPaw 版本号。"""
    return {"version": __version__}


app.include_router(api_router, prefix="/api")

app.include_router(
    agent_app.router,
    prefix="/api/agent",
    tags=["agent"],
)

# 语音通道（Voice Channel）：
# 这些 Twilio 面向的端点直接挂在根路径下（不在 `/api/` 下）。
# - `POST /voice/incoming`
# - `WS /voice/ws`
# - `POST /voice/status-callback`
app.include_router(voice_router, tags=["voice"])

# 挂载控制台静态资源：
# - 单个静态文件（logo/icon）
# - 将 `/assets/*` 挂载为专用静态目录
# - SPA 回退路由（`/{full_path:path}` -> `index.html`）
if os.path.isdir(_CONSOLE_STATIC_DIR):
    _console_path = Path(_CONSOLE_STATIC_DIR)

    @app.get("/logo.png")
    def _console_logo():
        f = _console_path / "logo.png"
        if f.is_file():
            return FileResponse(f, media_type="image/png")

        raise HTTPException(status_code=404, detail="Not Found")

    @app.get("/copaw-symbol.svg")
    def _console_icon():
        f = _console_path / "copaw-symbol.svg"
        if f.is_file():
            return FileResponse(f, media_type="image/svg+xml")

        raise HTTPException(status_code=404, detail="Not Found")

    _assets_dir = _console_path / "assets"
    if _assets_dir.is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=str(_assets_dir)),
            name="assets",
        )

    @app.get("/{full_path:path}")
    def _console_spa(full_path: str):
        if _CONSOLE_INDEX and _CONSOLE_INDEX.exists():
            return FileResponse(_CONSOLE_INDEX)

        raise HTTPException(status_code=404, detail="Not Found")
