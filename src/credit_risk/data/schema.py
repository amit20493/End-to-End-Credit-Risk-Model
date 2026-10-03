"""Xente transaction schema used for training and inference."""

ID_COLUMNS = [
    "TransactionId",
    "BatchId",
    "AccountId",
    "SubscriptionId",
    "CustomerId",
]

CATEGORICAL_COLUMNS = [
    "ProductCategory",
    "ChannelId",
    "ProviderId",
    "ProductId",
    "CurrencyCode",
]

NUMERIC_COLUMNS = [
    "Amount",
    "Value",
    "CountryCode",
    "PricingStrategy",
    "FraudResult",
]

TIME_COLUMN = "TransactionStartTime"
AMOUNT_COLUMN = "Amount"
CUSTOMER_COLUMN = "CustomerId"
TARGET_COLUMN = "is_high_risk"

REQUIRED_COLUMNS = (
    ID_COLUMNS
    + CATEGORICAL_COLUMNS
    + NUMERIC_COLUMNS
    + [TIME_COLUMN]
)

ENGINEERED_NUMERIC = [
    "TransactionHour",
    "TransactionDay",
    "TransactionMonth",
    "TransactionYear",
    "TotalTransactionAmount",
    "AverageTransactionAmount",
    "TransactionCount",
    "StdTransactionAmount",
]
