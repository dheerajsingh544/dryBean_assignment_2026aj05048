from pathlib import Path
import json, joblib, pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score,roc_auc_score,precision_score,recall_score,f1_score,matthews_corrcoef
P=Path(__file__).resolve().parent; M=P/"model"; M.mkdir(exist_ok=True)
def main():
 d=fetch_ucirepo(id=602); X=d.data.features.copy().apply(pd.to_numeric,errors="raise"); y=d.data.targets.iloc[:,0].astype(str)
 xt,xv,yt,yv=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
 test=xv.copy(); test["Class"]=yv.values; test.to_csv(P/"test_data.csv",index=False)
 models={
 "Logistic Regression":Pipeline([("scale",StandardScaler()),("clf",LogisticRegression(max_iter=3000))]),
 "Decision Tree":DecisionTreeClassifier(random_state=42),
 "K-Nearest Neighbors":Pipeline([("scale",StandardScaler()),("clf",KNeighborsClassifier(n_neighbors=5,weights="distance"))]),
 "Gaussian Naive Bayes":Pipeline([("scale",StandardScaler()),("clf",GaussianNB())]),
 "Support Vector Machine":Pipeline([("scale",StandardScaler()),("clf",CalibratedClassifierCV(SVC(C=2,kernel="rbf",random_state=42),ensemble=False))])}
 fn={"Logistic Regression":"logistic_regression","Decision Tree":"decision_tree","K-Nearest Neighbors":"knearest_neighbors","Gaussian Naive Bayes":"gaussian_naive_bayes","Support Vector Machine":"support_vector_machine"}
 rows=[]
 for n,m in models.items():
  print("Training",n); m.fit(xt,yt); pr=m.predict(xv); pb=m.predict_proba(xv)
  rows.append({"Model":n,"Accuracy":accuracy_score(yv,pr),"AUC":roc_auc_score(yv,pb,multi_class="ovr",average="weighted",labels=m.classes_),"Precision":precision_score(yv,pr,average="weighted",zero_division=0),"Recall":recall_score(yv,pr,average="weighted",zero_division=0),"F1":f1_score(yv,pr,average="weighted",zero_division=0),"MCC":matthews_corrcoef(yv,pr)})
  joblib.dump(m,M/(fn[n]+".joblib"))
 pd.DataFrame(rows).to_csv(P/"evaluation_metrics.csv",index=False)
 (P/"evaluation_metrics.json").write_text(json.dumps(rows,indent=2))
 (P/"dataset_info.json").write_text(json.dumps({"dataset":"UCI Dry Bean","uci_id":602,"features":list(X.columns),"target":"Class","classes":sorted(y.unique()),"instances":len(X),"train_rows":len(xt),"test_rows":len(xv),"random_state":42},indent=2))
 print(pd.DataFrame(rows).round(4).to_string(index=False))
if __name__=="__main__": main()
