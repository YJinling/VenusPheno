import logging
import os
import datetime
import torch

def get_logger(save_dir, name="train"):
    """
    创建 logger 对象（文件 + 终端双输出）
    """
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, f"{name}.log")

    # 如果已有 handler，要清空，否则重复打印
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
    logging.getLogger("").addHandler(console)
    logger = logging.getLogger(name)
    return logger


def log_model_info(logger, model, device="cuda"):
    """
    打印模型参数量与显存信息
    """
    # --- 参数统计 ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"🧠 Model Summary")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # --- 显存统计（仅GPU） ---
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        mem_alloc = torch.cuda.memory_allocated() / 1024**2
        mem_peak = torch.cuda.max_memory_allocated() / 1024**2
        logger.info(f"Current GPU Memory: {mem_alloc:.2f} MB")
        logger.info(f"Peak GPU Memory: {mem_peak:.2f} MB")
    else:
        logger.info("GPU not available or using CPU mode.")


def log_training_time(logger, start_time, end_time):
    """
    记录训练耗时
    """
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"⏱️ Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
