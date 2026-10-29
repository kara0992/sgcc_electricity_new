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
from data_fetcher import DataFetcher, RiskControlError, CredentialError

# 国网风控(RK001)处理参数默认值，main() 中会按环境变量覆盖
# 说明：不再做 6 小时长退避，恢复原来的短间隔重试节奏；
#      被 RK001 拦截最多尝试 RK001_MAX_RETRY 次，之后停止登录，等下一次计划任务。
RK001_MAX_RETRY = 4
# 两次尝试之间的等待秒数（默认 60 秒，设 0 表示不额外等待）
RK001_RETRY_WAIT_SECONDS = 60
# 同一账号连续被 RK001 拦截多少次后，切换到备用账号（邮箱/第二账号）
RK001_SWITCH_AFTER = 2

def main():
    global RETRY_TIMES_LIMIT, RK001_MAX_RETRY
    global RK001_RETRY_WAIT_SECONDS, RK001_SWITCH_AFTER
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
        # 遇到国网风控(RK001)：短间隔重试，最多 RK001_MAX_RETRY 次后停止登录
        RK001_MAX_RETRY = int(os.getenv("RK001_MAX_RETRY", 4))
        RK001_RETRY_WAIT_SECONDS = int(os.getenv("RK001_RETRY_WAIT_SECONDS", 60))
        # 备用账号（可为邮箱/用户名/第二个手机号）——主账号被风控时自动切换
        BACKUP_USERNAME = (os.getenv("BACKUP_USERNAME") or "").strip()
        BACKUP_PASSWORD = os.getenv("BACKUP_PASSWORD") or ""
        RK001_SWITCH_AFTER = int(os.getenv("RK001_SWITCH_AFTER", 2))

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
    fetcher = DataFetcher(PHONE_NUMBER, PASSWORD, BACKUP_USERNAME, BACKUP_PASSWORD)
    updator = SensorUpdator()

    # 生成随机延迟时间（-10分钟到+10分钟）
    random_delay_minutes = random.randint(-10, 10)
    parsed_time = datetime.strptime(JOB_START_TIME, "%H:%M") + timedelta(minutes=random_delay_minutes)
    logging.info(
        f"风控策略：被 RK001 拦截时不长时间等待，按 {RK001_RETRY_WAIT_SECONDS} 秒间隔重试，"
        f"最多尝试 {RK001_MAX_RETRY} 次；达到上限即停止登录，等下一次计划任务。")
    if BACKUP_USERNAME:
        logging.info(
            f"多账号策略：同一账号连续被 RK001 拦截 {RK001_SWITCH_AFTER} 次后，"
            f"自动切换到备用账号（{DataFetcher._mask_account(BACKUP_USERNAME)}）。")
    else:
        logging.info("未配置备用账号(BACKUP_USERNAME)，RK001 时只在主账号上重试。")
    logging.info(f"当前登录用户名为 {PHONE_NUMBER}，Home Assistant 地址为 {HASS_URL}，程序将每天在 {parsed_time.strftime('%H:%M')} 执行。")

    # 添加随机延迟
    next_run_time = parsed_time + timedelta(hours=12)

    logging.info(f'立即执行任务！下次运行时间为每天 {parsed_time.strftime("%H:%M")} 和 {next_run_time.strftime("%H:%M")}')
    schedule.every().day.at(parsed_time.strftime("%H:%M")).do(run_task, fetcher)
    schedule.every().day.at(next_run_time.strftime("%H:%M")).do(run_task, fetcher)

    # 重发缓存到 HA 的间隔（分钟），默认 60 分钟，用于防止 HA 重启后数据丢失
    CACHE_REPUBLISH_INTERVAL = int(os.getenv("CACHE_REPUBLISH_INTERVAL", "60"))
    schedule.every(CACHE_REPUBLISH_INTERVAL).minutes.do(republish_or_fetch, updator, fetcher)

    # 启动先把缓存里的“最近几天”日用电序列补发一次：
    # 不发新登录，也能让集成把之前漏掉的日期补齐（如 10-24/10-25）
    publish_cached_daily_series(fetcher)

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


def publish_cached_daily_series(fetcher: DataFetcher):
    """用本地缓存把“最近几天”的日用电序列补发一次（不触发新登录）。

    本地缓存 sgcc_cache.json 里保存着上次成功抓取的 tou_data（逐日总量与
    谷/平/峰/尖）。启动或每轮重发缓存时顺带推一条序列传感器，这样即使本次
    不发登录，集成也能把之前漏掉的日期补齐。
    """
    try:
        cache_file = os.path.join(get_data_dir(), "sgcc_cache.json")
        if not os.path.exists(cache_file):
            return
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for user_id, entry in (data or {}).items():
            tou = (entry or {}).get("tou_data") or {}
            if tou.get("daily"):
                fetcher._publish_daily_series(user_id, tou)
    except Exception as e:
        logging.warning(f"补发日用电序列失败(不影响主流程): {e}")


def republish_or_fetch(updator: SensorUpdator, fetcher: DataFetcher):
    if not updator.republish():
        logging.info("缓存数据已过期或不存在，正在从国家电网获取数据...")
        run_task(fetcher)
    else:
        # 缓存重发成功，顺带补发一次日用电序列（用于补齐历史空缺日期）
        publish_cached_daily_series(fetcher)


def run_task(data_fetcher: DataFetcher):
    """执行一次抓取任务。

    普通异常：沿用原有逻辑，立即重试 RETRY_TIMES_LIMIT 次。
    国网风控(RK001)：不再做 6 小时长退避，按 RK001_RETRY_WAIT_SECONDS 间隔重试，
    最多 RK001_MAX_RETRY 次；同一账号连续被拦 RK001_SWITCH_AFTER 次后换备用账号；
    达到次数上限仍失败则停止登录，等下一次计划任务。
    """
    err_retry = 0        # 普通异常重试次数（沿用 RETRY_TIMES_LIMIT）
    rk_attempt = 0       # 本次任务里 RK001 的尝试次数
    rounds_on_account = 0  # 当前账号上连续被 RK001 的次数（用于触发换号）

    while True:
        try:
            data_fetcher.fetch()
            if rk_attempt:
                logging.info(
                    f"第 {rk_attempt} 次重试后登录成功"
                    f"（账号 {data_fetcher.current_account_label()}），数据已恢复更新。")
            return
        except RuntimeError as e:
            # LLM 配置错误等不可恢复错误，立即退出
            logging.error(f"致命错误，程序退出: {e}")
            sys.exit(1)
        except CredentialError as e:
            # 账号/密码错误：绝不重试（5 次错误会锁号 20 分钟），直接停
            logging.error(
                f"[凭证错误] {e}；本次任务终止，请先核对账号与密码"
                f"（或该邮箱是否已绑定为该国网账号的登录名）。")
            return
        except RiskControlError as e:
            rk_attempt += 1
            rounds_on_account += 1
            logging.warning(
                f"[RK001] 第 {rk_attempt}/{RK001_MAX_RETRY} 次登录被国网风控拦截：{e}")
            if rk_attempt >= RK001_MAX_RETRY:
                logging.error(
                    f"已连续 {RK001_MAX_RETRY} 次登录被国网风控(RK001)拦截，"
                    f"本次停止登录尝试（不再重试），等下一次计划任务；"
                    f"当前账号 {data_fetcher.current_account_label()}。")
                return

            # 同一账号连续被拦达到阈值 → 换备用账号
            if (data_fetcher.has_backup_account()
                    and RK001_SWITCH_AFTER > 0
                    and rounds_on_account >= RK001_SWITCH_AFTER):
                if data_fetcher.switch_account(
                        f"账号 {data_fetcher.current_account_label()} 已连续 "
                        f"{rounds_on_account} 次被 RK001 风控"):
                    rounds_on_account = 0

            wait_seconds = RK001_RETRY_WAIT_SECONDS * (1 + random.uniform(0, 0.05))
            if wait_seconds > 0:
                logging.warning(
                    f"等待 {int(wait_seconds)} 秒后进行第 {rk_attempt + 1} 次尝试"
                    f"（账号 {data_fetcher.current_account_label()}）。")
                time.sleep(wait_seconds)
        except Exception as e:
            err_retry += 1
            logging.error(
                f"状态刷新任务失败，原因是 [{e}]，"
                f"还剩 {RETRY_TIMES_LIMIT - err_retry} 次重试机会。")
            if err_retry >= RETRY_TIMES_LIMIT:
                logging.error("已达到最大重试次数，本次任务结束，等下一次计划任务。")
                return


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
