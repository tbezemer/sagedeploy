from sagedeploy.model import load_model, predict_unseen
import logging

logger = logging.getLogger()

def test_predict_dry():
    passengers = [
        {'PassengerId': 1, 'Survived': 0, 'Pclass': 3, 'Name': 'Braund, Mr. Owen Harris', 'Sex': 'male', 'Age': 22.0,
         'SibSp': 1, 'Parch': 0, 'Ticket': 'A/5 21171', 'Fare': 7.25, 'Cabin': 'unknown', 'Embarked': 'S'},
        {'PassengerId': 2, 'Survived': 1, 'Pclass': 1, 'Name': 'Cumings, Mrs. John Bradley (Florence Briggs Thayer)',
         'Sex': 'female', 'Age': 38.0, 'SibSp': 1, 'Parch': 0, 'Ticket': 'PC 17599', 'Fare': 71.2833, 'Cabin': 'C85',
         'Embarked': 'C'},
        {'PassengerId': 3, 'Survived': 1, 'Pclass': 3, 'Name': 'Heikkinen, Miss. Laina', 'Sex': 'female', 'Age': 26.0,
         'SibSp': 0, 'Parch': 0, 'Ticket': 'STON/O2. 3101282', 'Fare': 7.925, 'Cabin': 'unknown', 'Embarked': 'S'},
        {'PassengerId': 4, 'Survived': 1, 'Pclass': 1, 'Name': 'Futrelle, Mrs. Jacques Heath (Lily May Peel)',
         'Sex': 'female', 'Age': 35.0, 'SibSp': 1, 'Parch': 0, 'Ticket': '113803', 'Fare': 53.1, 'Cabin': 'C123',
         'Embarked': 'S'},
        {'PassengerId': 5, 'Survived': 0, 'Pclass': 3, 'Name': 'Allen, Mr. William Henry', 'Sex': 'male', 'Age': 35.0,
         'SibSp': 0, 'Parch': 0, 'Ticket': '373450', 'Fare': 8.05, 'Cabin': 'unknown', 'Embarked': 'S'},
        {'PassengerId': 6, 'Survived': 0, 'Pclass': 3, 'Name': 'Moran, Mr. James', 'Sex': 'male', 'Age': 'unknown',
         'SibSp': 0, 'Parch': 0, 'Ticket': '330877', 'Fare': 8.4583, 'Cabin': 'unknown', 'Embarked': 'Q'},
        {'PassengerId': 7, 'Survived': 0, 'Pclass': 1, 'Name': 'McCarthy, Mr. Timothy J', 'Sex': 'male', 'Age': 54.0,
         'SibSp': 0, 'Parch': 0, 'Ticket': '17463', 'Fare': 51.8625, 'Cabin': 'E46', 'Embarked': 'S'},
        {'PassengerId': 8, 'Survived': 0, 'Pclass': 3, 'Name': 'Palsson, Master. Gosta Leonard', 'Sex': 'male',
         'Age': 2.0, 'SibSp': 3, 'Parch': 1, 'Ticket': '349909', 'Fare': 21.075, 'Cabin': 'unknown', 'Embarked': 'S'},
        {'PassengerId': 9, 'Survived': 1, 'Pclass': 3, 'Name': 'Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)',
         'Sex': 'female', 'Age': 27.0, 'SibSp': 0, 'Parch': 2, 'Ticket': '347742', 'Fare': 11.1333, 'Cabin': 'unknown',
         'Embarked': 'S'},
        {'PassengerId': 10, 'Survived': 1, 'Pclass': 2, 'Name': 'Nasser, Mrs. Nicholas (Adele Achem)', 'Sex': 'female',
         'Age': 14.0, 'SibSp': 1, 'Parch': 0, 'Ticket': '237736', 'Fare': 30.0708, 'Cabin': 'unknown', 'Embarked': 'C'}]
    pipeline = load_model('optml/model/model.pkl')
    predictions = predict_unseen(passengers, pipeline)

    logger.debug(str(predictions))
    return True
