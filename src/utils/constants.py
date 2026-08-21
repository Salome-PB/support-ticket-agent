class TicketCategory:
    TECHNICAL = "technical"
    BILLING = "billing"
    GENERAL = "general"
    ACCOUNT = "account"
    FEATURE_REQUEST = "feature_request"


class Priority:
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class Department:
    TECHNICAL_SUPPORT = "technical_support"
    BILLING_SUPPORT = "billing_support"
    CUSTOMER_SUCCESS = "customer_success"
    PRODUCT_TEAM = "product_team"
    ESCALATION_TEAM = "escalation_team"


class TicketStatus:
    NEW = "new"
    PROCESSING = "processing"
    APPROVED_NOT_SENT = "approved_not_sent"
    ESCALATED_NOT_SENT = "escalated_not_sent"
    REFUSED_NOT_SENT = "refused_not_sent"
    ASK_MORE_INFO_NOT_SENT = "ask_more_info_not_sent"


ESCALATION_KEYWORDS = [
    "urgent", "critical", "emergency", "asap", "immediately",
    "lawsuit", "legal", "attorney", "lawyer", "sue",
    "cancel", "refund", "money back", "charge back",
    "angry", "frustrated", "disappointed", "terrible",
]
