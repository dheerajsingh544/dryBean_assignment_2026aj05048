from pathlib import Path
import json,joblib,pandas as pd,streamlit as st,matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,roc_auc_score,precision_score,recall_score,f1_score,matthews_corrcoef,confusion_matrix,classification_report,ConfusionMatrixDisplay
P=Path(__file__).resolve().parent; TARGET="Class"
FILES={"Logistic Regression":"logistic_regression.joblib","Decision Tree":"decision_tree.joblib","K-Nearest Neighbors":"knearest_neighbors.joblib","Gaussian Naive Bayes":"gaussian_naive_bayes.joblib","Support Vector Machine":"support_vector_machine.joblib"}
st.set_page_config(page_title="Dry Bean Classifier",page_icon="🫘",layout="wide")
st.title("🫘 Dry Bean Variety Classification"); st.caption("M.Tech AIML • Machine Learning Assignment 2")
st.write("Compare five saved classifiers using labelled test data. Uploaded data is evaluated only; models are not retrained.")
@st.cache_resource
def load():
 models={}; missing=[]
 for n,f in FILES.items():
  p=P/"model"/f
  if p.exists(): models[n]=joblib.load(p)
  else: missing.append(f)
 return models,missing
models,missing=load()
if missing: st.error("Missing model files: "+", ".join(missing)); st.info("Run python train_models.py first."); st.stop()
info=json.loads((P/"dataset_info.json").read_text()); features=info["features"]
up=st.sidebar.file_uploader("Upload labelled test CSV",type="csv")
if up is None:
 if not (P/"test_data.csv").exists(): st.warning("Run train_models.py first."); st.stop()
 df=pd.read_csv(P/"test_data.csv"); st.sidebar.success("Using test_data.csv")
else:
 try: df=pd.read_csv(up)
 except Exception as e: st.error(str(e)); st.stop()
if TARGET not in df: st.error("CSV must contain target column Class."); st.stop()
missing_cols=sorted(set(features)-set(df.columns)); extra=sorted(set(df.columns)-set(features)-{TARGET})
if missing_cols or extra: st.error(f"Feature mismatch. Missing={missing_cols}; unexpected={extra}"); st.stop()
X=df[features].apply(pd.to_numeric,errors="coerce"); y=df[TARGET].astype(str)
if X.isna().any().any() or y.isna().any(): st.error("Missing or non-numeric values found."); st.stop()
a,b,c=st.columns(3); a.metric("Test instances",len(df)); b.metric("Features",len(features)); c.metric("Classes",y.nunique())
with st.expander("Preview test CSV"): st.dataframe(df.head(10),use_container_width=True)
def evaluate(m):
 p=m.predict(X); prob=m.predict_proba(X)
 return p,{"Accuracy":accuracy_score(y,p),"AUC":roc_auc_score(y,prob,multi_class="ovr",average="weighted",labels=m.classes_),"Precision":precision_score(y,p,average="weighted",zero_division=0),"Recall":recall_score(y,p,average="weighted",zero_division=0),"F1":f1_score(y,p,average="weighted",zero_division=0),"MCC":matthews_corrcoef(y,p)}
preds={}; rows=[]
for n,m in models.items():
 try: p,s=evaluate(m); preds[n]=p; rows.append({"Model":n,**s})
 except Exception as e: st.error(f"{n}: {e}"); st.stop()
comp=pd.DataFrame(rows).set_index("Model")
st.subheader("All-model comparison"); st.dataframe(comp.style.format("{:.4f}"),use_container_width=True); st.bar_chart(comp[["Accuracy","AUC","F1","MCC"]])
name=st.selectbox("Inspect model",list(models)); vals=comp.loc[name]; pred=preds[name]
st.subheader(name+" — detailed evaluation")
cols=st.columns(6)
for col,key in zip(cols,["Accuracy","AUC","Precision","Recall","F1","MCC"]): col.metric(key,f"{vals[key]:.4f}")
l,r=st.columns(2)
with l:
 fig,ax=plt.subplots(figsize=(7,5)); labels=sorted(set(y)|set(models[name].classes_))
 ConfusionMatrixDisplay(confusion_matrix(y,pred,labels=labels),display_labels=labels).plot(ax=ax,xticks_rotation=45,colorbar=False)
 fig.tight_layout(); st.pyplot(fig); plt.close(fig)
with r: st.dataframe(pd.DataFrame(classification_report(y,pred,zero_division=0,output_dict=True)).transpose().round(4),use_container_width=True)
st.download_button("Download comparison CSV",comp.reset_index().to_csv(index=False).encode(),"model_comparison.csv","text/csv")
st.markdown("---")
st.markdown("<div style='text-align:center;color:grey;font-size:0.85rem'>BITS Pilani — M.Tech (AIML) &nbsp;|&nbsp; Machine Learning Assignment 2<br><b>Dheeraj Singh</b> &nbsp;|&nbsp; ID: 2026AJ05048</div>",unsafe_allow_html=True)
