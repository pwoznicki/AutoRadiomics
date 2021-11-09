class Config:
    def __init__(self):
        self.name = "Radiomics"
        self.version = "0.1.0"
        self.author = "Piotr Woznicki"
        self.author_email = "piotrekwoznicki@gmail.com"
        self.available_classifiers = [
            "Random Forest",
            "AdaBoost",
            "Logistic Regression",
            "SVM",
            "Gaussian Process Classifier",
            "XGBoost",
        ]
