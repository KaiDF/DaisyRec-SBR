import logging

ACC_KPI = ['ndcg', 'mrr', 'hr']
LOG_DIR = 'seren/log/'

def get_logger(file_name):
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.DEBUG)

    # set two handlers
    log_file = LOG_DIR+file_name + '.log'

    fileHandler = logging.FileHandler(log_file, mode = 'w')
    fileHandler.setLevel(logging.DEBUG)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)

    # set formatter
    formatter = logging.Formatter('[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    consoleHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    # add
    logger.addHandler(fileHandler)
    logger.addHandler(consoleHandler)

    # logger.info("test")

    return logger