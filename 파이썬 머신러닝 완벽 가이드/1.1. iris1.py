import sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# 데이터 세트 로딩
iris = load_iris()
iris_data = iris.data
iris_label = iris.target

iris_data
iris_label

# DataFrame으로 변환
iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)
iris_df['label'] = iris.target
iris_df.head(3)

# 학습/검증 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_label, test_size=0.2, random_state=271)

# 결정트리 객체 생성
dt_clf = DecisionTreeClassifier(random_state=271)

# 학습 수행
dt_clf.fit(x_train, y_train)

# 예측
pred = dt_clf.predict(x_test)

# 평가
from sklearn.metrics import accuracy_score
print('예측 정확도: {0:.4f}'.format(accuracy_score(y_test, pred)))

