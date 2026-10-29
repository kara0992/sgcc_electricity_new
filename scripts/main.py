import nest_asyncio; nest_asyncio.apply()
import logging
import logging.config
import os
import sys
import time
import schedule
import json
import random
from error_watcher import ErrorWatcher
from sensor_updator import SensorUpdator

from datetime import datetime,timedelta
from const import *
from data_fetcher import DataFetcher, RiskControlError

# 国网风控(RK001)退避参数默认值，main() 中会按环境变量覆盖
RK001_WAIT_HOURS = 6.0
RK001_MAX_RETRY = 4

def main():
    global RETRY_TIMES_LIMIT, RK001_WAIT_HOURS, RK001_MAX_RETRY
    if 'PYTHON_IN_DOCKER' not in os.environ:
        # 读取 .env 文件
        import dotenv
        dotenv.load_dotenv(verbose=True)
    if os.path.isfile('/data/options.json'):
        with open('/data/options.json') as f:
            options = json.load(f)
        try:
            for key, value in options.items():
                os.environ[key] = str(value)
            import const
            const.LLM_API_KEY = os.getenv('LLM_API_KEY', '').strip()
            const.LLM_BASE_URL = os.getenv('LLM_BASE_URL', 'https://api.siliconflow.cn/v1')
            const.LLM_MODEL = os.getenv('LLM_MODEL', 'Qwen/Qwen3.5-35B-A3B')
            logging.info(f"当前以Homeassistant Add-on 形式运行.")
        except Exception as e:
            logging.error(f"读取 options.json 文件失败，程序将退出，错误信息: {e}。")
            sys.exit()

    try:
        PHONE_NUMBER = os.getenv("PHONE_NUMBER")
        logging.info(f"读取环境变量 PHONE_NUMBER : {PHONE_NUMBER}")
        PASSWORD = os.getenv("PASSWORD")
        HASS_URL = os.getenv("HASS_URL")
        JOB_START_TIME = os.getenv("JOB_START_TIME","07:00" ).strip('"').strip("'")
        LOG_LEVEL = os.getenv("LOG_LEVEL","INFO")
        VERSION = os.getenv("VERSION")
        RETRY_TIMES_LIMIT = int(os.getenv("RETRY_TIMES_LIMIT", 5))
        # 遇到国网风控(RK001)时的退避策略：等待若干小时后再试，最多重试若干轮
        RK001_WAIT_HOURS = float(os.getenv("RK001_WAIT_HOURS", 6))
        RK001_MAX_RETRY = int(os.getenv("RK001_MAX_RETRY", 4))

        logger_init(LOG_LEVEL)
        logging.info(f"当前以Docker镜像方式运行。")
    except Exception as e:
        logging.error(f"读取 .env 文件失败，程序将退出，错误信息: {e}。")
        sys.exit()

    logging.info(f"当前仓库版本为 {VERSION}，仓库地址为 https://github.com/ARC-MX/sgcc_electricity_new.git")
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"当前日期为 {current_datetime}。")

    logging.info(f"开始初始化 ErrorWatcher")
    ErrorWatcher.init(root_dir='/data/errors')
    logging.info(f'ErrorWatcher 初始化完成！')
    fetcher = DataFetcher(PHONE_NUMBER, PASSWORD)
    updator = SensorUpdator()

    # 生成随机延迟时间（-10分钟到+10分钟）
    random_delay_minutes = random.randint(-10, 10)
    parsed_time = datetime.strptime(JOB_START_TIME, "%H:%M") + timedelta(minutes=random_delay_minutes)
    logging.info(f"风控退避策略：遇到 RK001 时等待 {RK001_WAIT_HOURS:g} 小时后再试，最多 {RK001_MAX_RETRY} 轮。")
    logging.info(f"当前登录用户名为 {PHONE_NUMBER}，Home Assistant 地址为 {HASS_URL}，程序将每天在 {parsed_time.strftime('%H:%M')} 执行。")

    # 添加随机延迟
    next_run_time = parsed_time + timedelta(hours=12)

    logging.info(f'立即执行任务！下次运行时间为每天 {parsed_time.strftime("%H:%M")} 和 {next_run_time.strftime("%H:%M")}')
    schedule.every().day.at(parsed_time.strftime("%H:%M")).do(run_task, fetcher)
    schedule.every().day.at(next_run_time.strftime("%H:%M")).do(run_task, fetcher)

    # 重发缓存到 HA 的间隔（分钟），默认 60 分钟，用于防止 HA 重启后数据丢失
    CACHE_REPUBLISH_INTERVAL = int(os.getenv("CACHE_REPUBLISH_INTERVAL", "60"))
    schedule.every(CACHE_REPUBLISH_INTERVAL).minutes.do(republish_or_fetch, updator, fetcher)

    # 启动时先尝试从缓存恢复
    # 如果缓存恢复成功，则跳过本次启动时的实时抓取，避免频繁重启导致账号被封
    if not updator.republish():
        logging.info("未找到有效缓存，正在从国家电网获取数据...")
        run_task(fetcher)
    else:
        logging.info("已从缓存恢复数据，跳过启动时抓取以保护账号。")

    while True:
        schedule.run_pending()
        time.sleep(1)


def republish_or_fetch(updator: SensorUpdator, fetcher: DataFetcher):
    if not updator.republish():
        logging.info("缓存数据已过期或不存在，正在从国家电网获取数据...")
        run_task(fetcher)


def _sleep_until_retry(total_seconds: float, tag: str, chunk_seconds: int = 1800):
    """分段睡眠并在日志中输出心跳，避免长时间静默无法判断程序是否还活着。"""
    logging.warning(
        f"{tag}：根据风控退避策略，暂停所有登录尝试，等待 "
        f"{total_seconds / 3600:.1f} 小时后自动重试。")
    remaining = float(total_seconds)
    while remaining > 0:
        step = min(chunk_seconds, remaining)
        time.sleep(step)
        remaining -= step
        if remaining > 0:
            logging.info(f"{tag}：仍在等待，剩余约 {remaining / 3600:.1f} 小时。")


def run_task(data_fetcher: DataFetcher):
    """执行一次抓取任务。

    普通异常：沿用原有逻辑，立即重试 RETRY_TIMES_LIMIT 次。
    国网风控(RK001)：立即重试毫无意义且会加重风控，改为等待
    RK001_WAIT_HOURS 小时后再试，最多 RK001_MAX_RETRY 轮。
    """
    rk_round = 0
    while True:
        hit_rk001 = False
        for retry_times in range(1, RETRY_TIMES_LIMIT + 1):
            try:
                data_fetcher.fetch()
                if rk_round:
                    logging.info(f"风控等待 {rk_round} 轮后登录成功，数据已恢复更新。")
                return
            except RuntimeError as e:
                # LLM 配置错误等不可恢复错误，立即退出
                logging.error(f"致命错误，程序退出: {e}")
                sys.exit(1)
            except RiskControlError as e:
                logging.warning(f"[RK001] 检测到国网风控，立即停止本次快速重试：{e}")
                hit_rk001 = True
                break
            except Exception as e:
                logging.error(f"状态刷新任务失败，原因是 [{e}]，还剩 {RETRY_TIMES_LIMIT - retry_times} 次重试机会。")
                continue

        if not hit_rk001:
            # 普通异常已用完快速重试次数，本轮结束，等下一次计划任务
            return

        rk_round += 1
        if rk_round > RK001_MAX_RETRY:
            logging.error(
                f"已连续 {RK001_MAX_RETRY} 轮因国网风控(RK001)等待后仍失败，"
                f"本次任务结束，等下一次计划任务；建议更换出口 IP 或再等更久。")
            return

        # 加一点随机抖动（0~5%），避免固定整点特征被风控识别
        wait_seconds = RK001_WAIT_HOURS * 3600 * (1 + random.uniform(0, 0.05))
        _sleep_until_retry(
            wait_seconds,
            f"国网风控(RK001) 第 {rk_round}/{RK001_MAX_RETRY} 轮退避")

def logger_init(level: str):
    logger = logging.getLogger()
    logger.setLevel(level.strip().strip('\'" '))  # 兼容 .env 中带引号和空格的 LOG_LEVEL
    # 清除已有 handler（避免 basicConfig 自动添加的默认 handler 导致日志重复）
    logger.handlers.clear()
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)
    format = logging.Formatter("%(asctime)s  [%(levelname)-8s] ---- %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(format)
    logger.addHandler(sh)


if __name__ == "__main__":
    main()
