import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

class ObesityPredictor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_and_explore_data(self, file_path):
        print("="*60)
        print("LOADING AND EXPLORING DATA")
        print("="*60)
        self.df = pd.read_csv(file_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"\nDataset info:")
        print(self.df.info())
        print(f"\nFirst 5 rows:")
        print(self.df.head())
        print(f"\nTarget variable distribution:")
        print(self.df['NObeyesdad'].value_counts())
        print(f"\nMissing values:")
        print(self.df.isnull().sum())
        self.create_exploratory_plots()
        
    def create_exploratory_plots(self):
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 3, 1)
        self.df['NObeyesdad'].value_counts().plot(kind='bar', rot=45)
        plt.title('Obesity Categories Distribution')
        plt.xlabel('Obesity Category')
        plt.ylabel('Count')
        plt.subplot(2, 3, 2)
        plt.hist(self.df['Age'], bins=20, alpha=0.7)
        plt.title('Age Distribution')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.subplot(2, 3, 3)
        plt.scatter(self.df['Height'], self.df['Weight'], alpha=0.6)
        plt.title('Height vs Weight')
        plt.xlabel('Height')
        plt.ylabel('Weight')
        plt.subplot(2, 3, 4)
        self.df['Gender'].value_counts().plot(kind='bar')
        plt.title('Gender Distribution')
        plt.xlabel('Gender')
        plt.ylabel('Count')
        plt.subplot(2, 3, 5)
        self.df['family_history_with_overweight'].value_counts().plot(kind='bar')
        plt.title('Family History with Overweight')
        plt.xlabel('Family History')
        plt.ylabel('Count')
        plt.subplot(2, 3, 6)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig('exploratory_analysis.png', dpi=300, bbox_inches='tight')
        # plt.show()  # Commented out to prevent script from hanging
        
    def preprocess_data(self):
        print("="*60)
        print("PREPROCESSING DATA")
        print("="*60)
        df_processed = self.df.copy()
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        categorical_cols = categorical_cols.drop('NObeyesdad')
        print(f"Categorical columns to encode: {list(categorical_cols)}")
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            self.label_encoders[col] = le
        target_le = LabelEncoder()
        df_processed['NObeyesdad'] = target_le.fit_transform(df_processed['NObeyesdad'])
        self.label_encoders['NObeyesdad'] = target_le
        X = df_processed.drop('NObeyesdad', axis=1)
        y = df_processed['NObeyesdad']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        self.feature_names = X.columns.tolist()
        
    def train_baseline_models(self):
        print("="*60)
        print("TRAINING BASELINE MODELS")
        print("="*60)
        baseline_models = {
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
        }
        for name, model in baseline_models.items():
            print(f"\nTraining {name}...")
            if name == 'Logistic Regression':
                model.fit(self.X_train_scaled, self.y_train)
                predictions = model.predict(self.X_test_scaled)
                cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)
            else:
                model.fit(self.X_train, self.y_train)
                predictions = model.predict(self.X_test)
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
            accuracy = accuracy_score(self.y_test, predictions)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'predictions': predictions
            }
            print(f"Accuracy: {accuracy:.4f}")
            print(f"CV Score: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
            
    def train_ensemble_models(self):
        print("="*60)
        print("TRAINING ENSEMBLE MODELS")
        print("="*60)
        print("\n1. Training Random Forest (Bagging)...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(self.X_train, self.y_train)
        rf_predictions = rf_model.predict(self.X_test)
        rf_cv_scores = cross_val_score(rf_model, self.X_train, self.y_train, cv=5)
        self.results['Random Forest'] = {
            'model': rf_model,
            'accuracy': accuracy_score(self.y_test, rf_predictions),
            'cv_mean': rf_cv_scores.mean(),
            'cv_std': rf_cv_scores.std(),
            'predictions': rf_predictions
        }
        print("\n2. Training AdaBoost (Boosting)...")
        ada_model = AdaBoostClassifier(n_estimators=100, random_state=42)
        ada_model.fit(self.X_train, self.y_train)
        ada_predictions = ada_model.predict(self.X_test)
        ada_cv_scores = cross_val_score(ada_model, self.X_train, self.y_train, cv=5)
        self.results['AdaBoost'] = {
            'model': ada_model,
            'accuracy': accuracy_score(self.y_test, ada_predictions),
            'cv_mean': ada_cv_scores.mean(),
            'cv_std': ada_cv_scores.std(),
            'predictions': ada_predictions
        }
        print("\n3. Training Gradient Boosting...")
        gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb_model.fit(self.X_train, self.y_train)
        gb_predictions = gb_model.predict(self.X_test)
        gb_cv_scores = cross_val_score(gb_model, self.X_train, self.y_train, cv=5)
        self.results['Gradient Boosting'] = {
            'model': gb_model,
            'accuracy': accuracy_score(self.y_test, gb_predictions),
            'cv_mean': gb_cv_scores.mean(),
            'cv_std': gb_cv_scores.std(),
            'predictions': gb_predictions
        }
        print("\n4. Training XGBoost...")
        xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
        xgb_model.fit(self.X_train, self.y_train)
        xgb_predictions = xgb_model.predict(self.X_test)
        xgb_cv_scores = cross_val_score(xgb_model, self.X_train, self.y_train, cv=5)
        self.results['XGBoost'] = {
            'model': xgb_model,
            'accuracy': accuracy_score(self.y_test, xgb_predictions),
            'cv_mean': xgb_cv_scores.mean(),
            'cv_std': xgb_cv_scores.std(),
            'predictions': xgb_predictions
        }
        print("\n5. Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        lgb_model.fit(self.X_train, self.y_train)
        lgb_predictions = lgb_model.predict(self.X_test)
        lgb_cv_scores = cross_val_score(lgb_model, self.X_train, self.y_train, cv=5)
        self.results['LightGBM'] = {
            'model': lgb_model,
            'accuracy': accuracy_score(self.y_test, lgb_predictions),
            'cv_mean': lgb_cv_scores.mean(),
            'cv_std': lgb_cv_scores.std(),
            'predictions': lgb_predictions
        }
        print("\n6. Training Voting Classifier...")
        voting_model = VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
                ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
                ('xgb', xgb.XGBClassifier(n_estimators=50, random_state=42))
            ],
            voting='soft'
        )
        voting_model.fit(self.X_train, self.y_train)
        voting_predictions = voting_model.predict(self.X_test)
        voting_cv_scores = cross_val_score(voting_model, self.X_train, self.y_train, cv=5)
        self.results['Voting Classifier'] = {
            'model': voting_model,
            'accuracy': accuracy_score(self.y_test, voting_predictions),
            'cv_mean': voting_cv_scores.mean(),
            'cv_std': voting_cv_scores.std(),
            'predictions': voting_predictions
        }
        print("\n7. Training Stacking Classifier...")
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
            ('xgb', xgb.XGBClassifier(n_estimators=50, random_state=42))
        ]
        stacking_model = StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(random_state=42),
            cv=5
        )
        stacking_model.fit(self.X_train, self.y_train)
        stacking_predictions = stacking_model.predict(self.X_test)
        stacking_cv_scores = cross_val_score(stacking_model, self.X_train, self.y_train, cv=3)
        self.results['Stacking Classifier'] = {
            'model': stacking_model,
            'accuracy': accuracy_score(self.y_test, stacking_predictions),
            'cv_mean': stacking_cv_scores.mean(),
            'cv_std': stacking_cv_scores.std(),
            'predictions': stacking_predictions
        }
        
    def evaluate_models(self):
        print("="*60)
        print("MODEL EVALUATION AND COMPARISON")
        print("="*60)
        results_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [self.results[model]['accuracy'] for model in self.results.keys()],
            'CV Mean': [self.results[model]['cv_mean'] for model in self.results.keys()],
            'CV Std': [self.results[model]['cv_std'] for model in self.results.keys()]
        })
        results_df = results_df.sort_values('Accuracy', ascending=False)
        print("\nModel Performance Summary:")
        print(results_df.to_string(index=False))
        results_df.to_csv('model_results.csv', index=False)
        self.create_comparison_plots(results_df)
        best_model_name = results_df.iloc[0]['Model']
        best_model = self.results[best_model_name]['model']
        best_predictions = self.results[best_model_name]['predictions']
        print(f"\n\nDetailed Evaluation of Best Model: {best_model_name}")
        print("="*50)
        target_names = self.label_encoders['NObeyesdad'].classes_
        print("\nClassification Report:")
        print(classification_report(self.y_test, best_predictions, target_names=target_names))
        cm = confusion_matrix(self.y_test, best_predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_names, yticklabels=target_names)
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        # plt.show()  # Commented out to prevent script from hanging
        # Feature importance for best model (if available)
        if hasattr(best_model, 'feature_importances_'):
            self.plot_feature_importance(best_model, best_model_name)
        
    def create_comparison_plots(self, results_df):
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        plt.barh(results_df['Model'], results_df['Accuracy'])
        plt.xlabel('Accuracy')
        plt.title('Model Accuracy Comparison')
        plt.gca().invert_yaxis()
        plt.subplot(2, 2, 2)
        plt.barh(results_df['Model'], results_df['CV Mean'])
        plt.xlabel('CV Mean Score')
        plt.title('Cross-Validation Mean Score Comparison')
        plt.gca().invert_yaxis()
        plt.subplot(2, 2, 3)
        plt.barh(results_df['Model'], results_df['CV Std'])
        plt.xlabel('CV Standard Deviation')
        plt.title('Cross-Validation Standard Deviation')
        plt.gca().invert_yaxis()
        plt.subplot(2, 2, 4)
        plt.scatter(results_df['CV Mean'], results_df['Accuracy'])
        for i, txt in enumerate(results_df['Model']):
            plt.annotate(txt, (results_df['CV Mean'].iloc[i], results_df['Accuracy'].iloc[i]), 
                        rotation=45, ha='left', va='bottom')
        plt.xlabel('CV Mean Score')
        plt.ylabel('Test Accuracy')
        plt.title('CV Mean vs Test Accuracy')
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        # plt.show()  # Commented out to prevent script from hanging
        
    def plot_feature_importance(self, model, model_name):
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            plt.figure(figsize=(10, 8))
            plt.barh(importance_df['Feature'], importance_df['Importance'])
            plt.xlabel('Feature Importance')
            plt.title(f'Feature Importance - {model_name}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            # plt.show()  # Commented out to prevent script from hanging
            importance_df.to_csv('feature_importance.csv', index=False)
            
    def run_complete_analysis(self, file_path):
        print("STARTING COMPLETE OBESITY PREDICTION ANALYSIS")
        print("="*60)
        self.load_and_explore_data(file_path)
        self.preprocess_data()
        self.train_baseline_models()
        self.train_ensemble_models()
        self.evaluate_models()
        # Manually generate feature importance for Random Forest
        self.plot_feature_importance(self.results['Random Forest']['model'], 'Random Forest')
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print("Files generated:")
        print("- exploratory_analysis.png")
        print("- model_comparison.png")
        print("- confusion_matrix.png")
        print("- feature_importance.png")
        print("- model_results.csv")
        print("- feature_importance.csv")

if __name__ == "__main__":
    predictor = ObesityPredictor()
    predictor.run_complete_analysis('dataset.csv')

