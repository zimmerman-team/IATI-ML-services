import sys
import os

"""
Contains the specification of the field mappings for IATI activity
"""

# FIXME: find a way around this sys path fix, maybe with __init__.py
sys.path.append(
    os.path.abspath(
        os.path.dirname(
            os.path.abspath(__file__)
        )+"/.."
    )
)

from common.relspecs_classes import *

"""
the specification for all relation fields is in the `rels` public module variable
"""
rels = RelsCollection([
    Rel("activity_date", [
        CategoryField("type","ActivityDateType"),
        DatetimeField("iso_date")
    ], download=True),
    Rel("budget", [
        CategoryField(
            "value_currency",
            'Currency',
            # output_activation_function=torch.nn.Sigmoid(),
            # loss_function=torch.nn.MSELoss(),
            prevent_constant_prediction='USD'
        ),
        CategoryField("type", 'BudgetType'),
        CategoryField("status", 'BudgetStatus'),
        DatetimeField("period_start_iso_date"),
        DatetimeField("period_end_iso_date"),
        NumericalField("value")
    ], download=True),
    Rel("result", [
        CategoryField("type", 'ResultType'),
        TextField("title_narrative"),
        # TextField("description_narrative"),
        # aggregation_status
        # CategoryField("indicator_measure","IndicatorMeasure"),
        # indicator_ascending
        # indicator_aggregation_status
        # TextField("indicator_title_narrative"),
        # CategoryField("indicator_title_narrative_lang","Language"),
        # TextField("indicator_description_narrative"),
        # CategoryField("indicator_description_narrative_lang","Language"),
        # NumericalField("indicator_baseline_year"),
        # DatetimeField("indicator_baseline_iso_date"),
        # indicator_baseline_value

        # FIXME TODO WARNING: following fields may be presented multiple times for each
        # result instance. Hence their cardinality may be k*cardinality(result)
        # should I consider only the first for each result?
        # But then they are not grouped for each result but all put in the same list,
        # so that might be difficult.
        # DatetimeField("indicator_period_period_start_iso_date"),
        # DatetimeField("indicator_period_period_end_iso_date"),
        # NumericalField("indicator_period_target_value"),
        # NumericalField("indicator_period_actual_value")
    ], download=True),
    Rel("participating_org", [
        TextField("ref"),
        CategoryField("type","OrganisationType"),
        CategoryField("role","OrganisationRole"),
        TextField("narrative"),
    ], download=True),
    Rel("contact_info", [
        TextField("email"),
        TextField("website"),
        TextField("organisation_narrative"),
    ], download=True),
    Rel("location", [
        TextField("ref"),
        CategoryField("reach_code", "GeographicLocationReach"),
        PositionField("point_pos"),
        CategoryField("exactness_code", "GeographicExactness"),
        CategoryField("class_code","GeographicLocationClass"),
        TextField("name_narrative")
    ], download=True),
    Rel("sector", [
        CategoryField("vocabulary","SectorVocabulary"),
        CategoryField("code","SectorCategory")
    ], download=True),
    Rel("policy_marker", [
        CategoryField("vocabulary","PolicyMarkerVocabulary"),
        CategoryField("code","PolicyMarker"),
        CategoryField("significance", "PolicySignificance")
    ], download=True),
    Rel("default_aid_type", [
        CategoryField("vocabulary","AidTypeVocabulary"),
        CategoryField("code","AidType")
    ], download=True),
    Rel("transaction", [
        TextField("ref"),
        BooleanField("humanitarian"),
        CategoryField("type","TransactionType"),
        DatetimeField("date_iso_date"),
        CategoryField("value_currency","Currency"),
        DatetimeField("value_date"),
        NumericalField("value"),
        NumericalField("value_usd"),
        TextField("provider_org_provider_activity_id"),
        CategoryField("provider_org_type","OrganisationType"),
        TextField("provider_org_ref"),
        CategoryField("flow_type_code","FlowType"),
        CategoryField("finance_type_code","FinanceType"),
        CategoryField("tied_status_code","TiedStatus")
    ],
        download=True,

        # airflow was unable to create very large numpy arrays from the transaction entity
        #   because it gets a very high dimensionality as there are a large number of fields.
        #   Hence we limit the amount of datapoints to the following amount.
        limit=100000
        )
])

"""
the specification for an activity excluding relation fields is in the `activity` public module variable
"""
activity = Activity("activity",[
    TextField("iati_identifier"),
    CategoryField("default_lang","Language"),
    CategoryField("default_currency","Currency"),
    BooleanField("humanitarian"),
    CategoryField("activity_status_code", "ActivityStatus"),
    CategoryField("collaboration_type_code", "CollaborationType"),
    #FIXME this field needs to be properly mapped: "hierarchy"
    CategoryField("default_flow_type", "FlowType"),
    CategoryField("default_finance_type_code", "FinanceType"),
    CategoryField("default_tied_status_code","TiedStatus")
], download=True)

"""
`spec` is a public list containing relation field specification and activity (excluding relation fields) specification
"""
specs = rels + [activity]
