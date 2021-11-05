import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import sys
import os
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.as_posix() # add project root path for jupyter/CLI
sys.path.insert(0, ROOT_DIR)
#sys.path.insert(0, "/Users/kaushik/MyStuff/Workspace/NEU/DS5500/Project/DS5500_CapstoneProject") # add project path for jupyter/CLI local
#sys.path.insert(0, "/home/kaushik/DS5500") # add project path for jupyter/CLI remote

from src.config import cfg
from src.data import prepOPPCorpus
from src.utils import gen
from src.main import driver





def main():

    """
    MAIN FUNCTION TO CALL DRIVERS
    """

    # Create dataset for modeling and VIZ
    prepOPPCorpus.createDataset(cfg, splitcat = False, metadata = False, relational_data = False)

    # Create global param dict for modeling
    params = gen.createParamDict(cfg)

    # Perform hyper-param tuning
    driver.performTuning(params, study_name="optimization", n_trials=5)

    # Train model with custom (best) params
    #params = gen.loadParams(os.path.join(cfg.PARAM.BEST_PARAM_DPATH, "best_param_dict.json"))
    #driver.trainwithBP(params, experiment_name="test", run_name="run1", save=True)

    # Predict segment with custom run_id
    #run_id = "3496df67988c4d1c990ed6f2a43016e8"
    text = "When You access the Service by or through a mobile device, We may collect certain information automatically, " \
          "including, but not limited to, the type of mobile device You use, Your mobile device unique ID, the IP address of" \
           " Your mobile device, Your mobile operating system, the type of mobile Internet browser You use, unique device identifiers " \
           "and other diagnostic data."
    #driver.predictSegment(text, run_id)

    # Get params of custom run_id
    #driver.getRunParams(run_id)

    # Get performance of custom run_id
    #driver.getRunMetrics(run_id)

    # Serve MLFlow UI
    # mlflow ui --backend-store-uri file:///Users/kaushik/MyStuff/Workspace/NEU/DS5500/Project/DS5500_CapstoneProject/mlflow_registry

    # MLflow kill process using port 5000 or CTRL+C
    #kill -9 $(lsof -i:5000 -t) 2> /dev/null

    # Delete test experiment
    #driver.deleteMLFlowExperiment(experiment_name="test")



if __name__ == '__main__':
    main()
