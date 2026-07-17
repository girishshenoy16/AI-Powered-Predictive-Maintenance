from src.preprocess import process_train_data, process_test_data
from src.model_trainer import train_and_evaluate

if __name__ == "__main__":
    print("=======================================")
    print(" AI Predictive Maintenance Data Pipeline")
    print("=======================================")
    
    train_df = process_train_data()
    test_df = process_test_data()
    
    train_and_evaluate(train_df, test_df)
