from feast import FeatureStore
from feast import (
    Entity,
    FeatureView,
    FileSource,
    Field,
    RequestSource,
    FeatureService,
    PushSource,
)
from feast.types import Int64, Float64, String
from feast.on_demand_feature_view import on_demand_feature_view

customer_marketing_campaign = Entity(name="customer_marketing_campaign", join_keys=["customer_marketing_campaign_id"])

customer_marketing_campaign_source = FileSource(
    name="customer_marketing_campaign_source",
    path="data/marketing_campaign_feature_table.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)


customer_marketing_campaign_push_source = PushSource(
    name="customer_marketing_campaign_push_source",
    batch_source=customer_marketing_campaign_source,
)

customer_marketing_campaign_view = FeatureView(
    name="customer_marketing_campaign_view",
    entities=[customer_marketing_campaign],
    online=True,
    schema=[
        Field(name="Customer_Age", dtype=Float64, description="Edad del cliente."),
        Field(name="Education", dtype=Float64, description="Nivel de educación del cliente."),
        Field(name="Income", dtype=Float64, description="Monto de ingresos anuales del cliente."),
        Field(name="Kidhome", dtype=Float64, description="Número de niños por casa familiar del cliente."),
        Field(name="Teenhome", dtype=Float64, description="Número de jóvenes por casa familiar del cliente."),
        Field(name="Customer_Tenure", dtype=Float64, description="Permanencia en días del cliente."),
        Field(name="Recency", dtype=Float64, description="Días transcurridos desde la última compra del cliente."),
        Field(name="MntWines", dtype=Float64, description="Monto gastado en vino por el cliente el último año."),
        Field(name="MntFruits", dtype=Float64, description="Monto gastado en frutas por el cliente el último año."),
        Field(name="MntMeatProducts", dtype=Float64, description="Monto gastado en carnes por el cliente el último año."),
        Field(name="MntFishProducts", dtype=Float64, description="Monto gastado en pescados por el cliente el último año."),
        Field(name="MntSweetProducts", dtype=Float64, description="Monto gastado en dulces por el cliente el último año."),
        Field(name="MntGoldProds", dtype=Float64, description="Monto gastado en joyas de oro por el cliente el último año."),
        Field(name="NumDealsPurchases", dtype=Float64, description="Número de compras realizadas por el cliente en el supermarket."),
        Field(name="NumWebPurchases", dtype=Float64, description="Número de compras realizadas por el cliente en la web del supermarket."),
        Field(name="NumCatalogPurchases", dtype=Float64, description="Número de compras realizadas por el cliente por catálogo del supermarket."),
        Field(name="NumStorePurchases", dtype=Float64, description="Número de compras realizadas por el cliente presencialmente en el supermarket."),
        Field(name="NumWebVisitsMonth", dtype=Float64, description="Número de visitas mensuales realizadas por el cliente a la web del supermarket."),
        Field(name="AcceptedCmp3", dtype=Float64, description="Flag binario que indica si el cliente realizó una compra durante la tercera campaña del supermarket."),
        Field(name="AcceptedCmp4", dtype=Float64, description="Flag binario que indica si el cliente realizó una compra durante la cuarta campaña del supermarket."),
        Field(name="AcceptedCmp5", dtype=Float64, description="Flag binario que indica si el cliente realizó una compra durante la quinta campaña del supermarket."),
        Field(name="AcceptedCmp1", dtype=Float64, description="Flag binario que indica si el cliente realizó una compra durante la primera campaña del supermarket."),
        Field(name="AcceptedCmp2", dtype=Float64, description="Flag binario que indica si el cliente realizó una compra durante la segunda campaña del supermarket."),
        Field(name="Complain", dtype=Float64, description="Flag binario que indica si el cliente realizó una queja del supermarket."),
        Field(name="Response", dtype=Float64, description="Flag binario que indica si el cliente realizó una compra durante la actual campaña del supermarket."),
        Field(name="Marital_Status_Alone", dtype=Float64, description="Estado civil del cliente."),
        Field(name="Marital_Status_Divorced", dtype=Float64, description="Estado civil del cliente."),
        Field(name="Marital_Status_Married", dtype=Float64, description="Estado civil del cliente."),
        Field(name="Marital_Status_Single", dtype=Float64, description="Estado civil del cliente."),
        Field(name="Marital_Status_Together", dtype=Float64, description="Estado civil del cliente."),
        Field(name="Marital_Status_Widow", dtype=Float64, description="Estado civil del cliente."),
        Field(name="Marital_Status_YOLO", dtype=Float64, description="Estado civil del cliente."),
        Field(name="created", dtype=String, description="Timestamp de creación de features."),
    ],
    source=customer_marketing_campaign_source,
)

input_request = RequestSource(
    name="input_request",
    schema=[
        Field(name="customer_marketing_campaign_id", dtype=String),
    ],
)

feature_service = FeatureService(
    name="customer_marketing_campaign_feature_service",
    features=[customer_marketing_campaign_view],
)