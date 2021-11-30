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
    #prepOPPCorpus.createDataset(cfg, splitcat = False, metadata = False, relational_data = False)

    # Define experiment name
    experiment_name = "CNN_W_FE_M"
    # Define run name for retraining
    run_name = "RETRAIN_BP"

    # Create global param dict for modeling
    #params = gen.createParamDict(cfg)

    # Perform hyper-param tuning
    #driver.performTuning(params, study_name = experiment_name, n_trials=50)

    # Train model with custom (best) params (leave as None to automatically infer best params)
    #params = gen.loadParams(os.path.join(cfg.PARAM.CUSTOM_PARAM_FPATH))
    #driver.trainwithBP(param_dict = params, experiment_name = experiment_name, run_name = run_name, save=True)

    # Predict segment with custom run_id
    experiment_dpath = os.path.join(cfg.PARAM.BEST_PARAM_DPATH, "best_params_" + experiment_name)
    run_id = gen.loadID(os.path.join(experiment_dpath, "run_ID.txt"))
    text = ["When You access the Service by or through a mobile device, We may collect certain information automatically, " \
           "including, but not limited to, the type of mobile device You use, Your mobile device unique ID, the IP address of" \
            " Your mobile device, Your mobile operating system, the type of mobile Internet browser You use, unique device identifiers " \
            "and other diagnostic data.", "When You access the Service by or through a mobile device, We may collect certain information automatically, " \
                                          "including, but not limited to, the type of mobile device You use, Your mobile device unique ID, the IP address of" \
                                          " Your mobile device, Your mobile operating system, the type of mobile Internet browser You use, unique device identifiers " \
                                          "and other diagnostic data."]
    segments_processed = ["If you use a Microsoft product with an account provided by an organization you are affiliated with, such as your work or school account, that organization can",
                          "Control and administer your Microsoft product and product account, including controlling privacy-related settings of the product or product account.",
                          "Access and process your data, including the interaction data, diagnostic data, and the contents of your communications and files associated with your Microsoft product and product accounts.",
                          "If you lose access to your work or school account (in event of change of employment, for example), you may lose access to products and the content associated with those products, including those you acquired on your own behalf, if you used your work or school account to sign in to such products.",
                          "Many Microsoft products are intended for use by organizations, such as schools and businesses. Please see the Enterprise and developer products section of this privacy statement. If your organization provides you with access to Microsoft products, your use of the Microsoft products is subject to your organization's policies, if any. You should direct your privacy inquiries, including any requests to exercise your data protection rights, to your organization’s administrator. When you use social features in Microsoft products, other users in your network may see some of your activity. To learn more about the social features and other functionality, please review documentation or help content specific to the Microsoft product. Microsoft is not responsible for the privacy or security practices of our customers, which may differ from those set forth in this privacy statement.",
                          "When you use a Microsoft product provided by your organization, Microsoft’s processing of your personal data in connection with that product is governed by a contract between Microsoft and your organization. Microsoft processes your personal data to provide the product to your organization and you, and in some cases for Microsoft’s business operations related to providing the product as described in the Enterprise and developer products section. As mentioned above, if you have questions about Microsoft’s processing of your personal data in connection with providing products to your organization, please contact your organization. If you have questions about Microsoft’s business operations in connection with providing products to your organization as provided in the Product Terms, please contact Microsoft as described in the How to contact us section. For more information on our business operations, please see the Enterprise and developer products section.","For Microsoft products provided by your K-12 school, including Microsoft 365 Education, Microsoft will:"]
    #driver.predictSegment(text, run_id)
    driver.productionPredict(segments_processed, run_id, multi_threshold = True)

    # Get params of custom run_id
    #driver.getRunParams(run_id)

    # Get performance of custom run_id
    #driver.getRunMetrics(run_id)

    # Serve MLFlow UI
    # mlflow ui --backend-store-uri file:///Users/kaushik/MyStuff/Workspace/NEU/DS5500/Project/DS5500_CapstoneProject/mlflow_registry

    # MLflow kill process using port 5000 or CTRL+C
    #kill -9 $(lsof -i:5000 -t) 2> /dev/null

    # Delete test experiment
    #driver.deleteMLFlowExperiment(experiment_name=experiment_name)



if __name__ == '__main__':
    main()
