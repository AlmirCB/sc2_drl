from codecarbon import OfflineEmissionsTracker
from codecarbon.output import LoggerOutput
import logging
import sys
import os
from codecarbon.core.util import count_cpus, suppress
from codecarbon.external.logger import logger


@suppress(Exception)
def flush(tracker):
    """
    Returns tracker emissions data. This is a modif over the original tracker "flush" method
    """
    if tracker._start_time is None:
        logger.error("You first need to start the tracker.")
        return None

    # Run to calculate the power used from last
    # scheduled measurement to shutdown
    tracker._measure_power_and_energy()

    emissions_data = tracker._prepare_emissions_data()
    # tracker._persist_data(emissions_data)

    return emissions_data

def get_codecarbon_tracker(out_folder):
    """Sets given out_folder to export output_file (emissions.json) 
    and logfile (codecarbon.log)"""

    output_file_name = "emissions"
    output_file_folder = os.path.join(out_folder, output_file_name + ".json")
    log_name = "codecarbon"
    log_folder = os.path.join(out_folder, log_name + ".log")

    
    """
    As emissions are being captured via "flush" custom method in this same file, don't need output file anymore
    """
    # # Create a dedicated logger (log name can be the CodeCarbon project name for example)
    # _logger = logging.getLogger(output_file_name)

    # # Add a handler, see Python logging for various handlers (here a local file named after log_name)
    # _channel = logging.FileHandler(output_file_folder)
    # _logger.addHandler(_channel)

    # # Set logging level from DEBUG to CRITICAL (typically INFO)
    # # This level can be used in the logging process to filter emissions messages
    # _logger.setLevel(logging.INFO)

    # # Create a CodeCarbon LoggerOutput with the logger, specifying the logging level to be used for emissions data messages
    # my_logger = LoggerOutput(_logger, logging.INFO)

    tracker = OfflineEmissionsTracker(
            country_iso_code="ESP",
            measure_power_secs=30,
            project_name="",
            output_dir=out_folder,
            # save_to_logger=True,
            # logging_logger=my_logger,
            save_to_file = False,
        )

    # Clear CodeCarbon handlers
    logger = logging.getLogger(log_name)
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])

    # Define a log formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)-12s: %(levelname)-8s %(message)s"
    )

    # Create file handler which logs debug messages
    fh = logging.FileHandler(log_folder)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    consoleHandler.setLevel(logging.WARNING)
    logger.addHandler(consoleHandler)

    logger.debug("GO!")

    return tracker

