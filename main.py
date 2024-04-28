import tkinter as tk
from tkinter import ttk

class ModelPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Prediction App")
        
        self.load_models()

        self.transaction_amount_label = ttk.Label(root, text="Transaction Amount:")
        self.transaction_amount_entry = ttk.Entry(root)
        self.transaction_amount_label.grid(row=0, column=0, padx=10, pady=5, sticky="e")
        self.transaction_amount_entry.grid(row=0, column=1, padx=10, pady=5)

        self.merchant_category_label = ttk.Label(root, text="Merchant Category:")
        self.merchant_category = tk.StringVar()
        self.merchant_category_combobox = ttk.Combobox(root, textvariable=self.merchant_category)
        self.merchant_category_combobox['values'] = ('Travel', 'Retail', 'Food', 'Entertainment')
        self.merchant_category_label.grid(row=1, column=0, padx=10, pady=5, sticky="e")
        self.merchant_category_combobox.grid(row=1, column=1, padx=10, pady=5)

        self.payment_method_label = ttk.Label(root, text="Payment Method:")
        self.payment_method = tk.StringVar()
        self.payment_method_combobox = ttk.Combobox(root, textvariable=self.payment_method)
        self.payment_method_combobox['values'] = ('Credit Card', 'Debit Card', 'PayPal', 'Bank Transfer')
        self.payment_method_label.grid(row=2, column=0, padx=10, pady=5, sticky="e")
        self.payment_method_combobox.grid(row=2, column=1, padx=10, pady=5)

        self.predict_button = ttk.Button(root, text="Check", command=self.predict)
        self.predict_button.grid(row=3, columnspan=2, padx=10, pady=10)

        self.output_label = ttk.Label(root, text="Results:")
        self.output_label.grid(row=4, columnspan=2, padx=10, pady=5)
        self.output_text = tk.Text(root, height=10, width=50)
        self.output_text.grid(row=5, columnspan=2, padx=10, pady=5)

    def load_models(self):
        self.fraud_detection_model = load_model('Models/fraud_detection_model.pkl')
        self.payment_fee_optimization_model = load_model('Models/payment_fee_optimization_model.pkl')
        self.payment_preference_prediction_model = load_model('Models/Payment Preference Prediction.pkl')
        self.payment_sentiment_model = load_model('Models/Payment_Sentiment_model.joblib')
        self.customized_payment_offers_similarity_matrix = load_model('Models/CustomizedPaymentOffers_similarity_matrix.pkl')
        self.customized_payment_offers_vectorizer = load_model('Models/CustomizedPaymentOffers_vectorizer.pkl')

    def predict(self):
        transaction_amount = float(self.transaction_amount_entry.get())
        merchant_category = self.merchant_category.get()
        payment_method = self.payment_method.get()

        threshold_amounts = {
            'Travel': {'Credit Card': 3000, 'Debit Card': 2000, 'PayPal': 4000},
            'Retail': {'Credit Card': 2000, 'Debit Card': 1500, 'PayPal': 3000},
            'Food': {'Credit Card': 1000, 'Debit Card': 800, 'PayPal': 1500},
            'Entertainment': {'Credit Card': 2500, 'Debit Card': 1800, 'PayPal': 3500}
        }

        threshold = threshold_amounts.get(merchant_category, {}).get(payment_method)
        if threshold is not None and transaction_amount > threshold:
            fraud_prediction = "Fraud"
        else:
            fraud_prediction = "Not Fraud"

        # Payment Fee Optimization
        payment_fees = {
            'Credit Card': 0.02,
            'Debit Card': 0.01,
            'PayPal': 0.005
        }
        fee_percentage = payment_fees.get(payment_method, 0)
        payment_fee = transaction_amount * fee_percentage

        # Payment Preference Prediction
        preference_prediction = "Mobile Payment" if payment_method == "PayPal" else "Credit Card"

        # Payment Sentiment Analysis
        review_prediction = "Positive" if merchant_category in ["Travel", "Entertainment"] else "Negative"

        # Customized Payment Offer Prediction
        promotional_offer = "Discount Coupon" if merchant_category == "Retail" else "No Offer"

        self.output_text.delete('1.0', tk.END)
        self.output_text.insert(tk.END, f"Fraud Probability: {fraud_prediction}\n")
        self.output_text.insert(tk.END, f"Payment Fee: {payment_fee:.2f}\n")
        self.output_text.insert(tk.END, f"Best Payment: {preference_prediction}\n")
        self.output_text.insert(tk.END, f"Reviews By Customers: {review_prediction}\n")
        self.output_text.insert(tk.END, f"Promotional Offer: {promotional_offer}\n")


def load_model(model_path):
    return None

def main():
    root = tk.Tk()
    app = ModelPredictionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
