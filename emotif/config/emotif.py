from __future__ import unicode_literals
from frappe import _
#
def get_data():
	return [
		{

			"color": "#1abc9c",
			"icon": "fa fa-check-square-o",
			"label": _("Emotif"),
			"type": "module",
			"onboard_present": 1,
            "items": [
                {
                    "type": "doctype",
                    "name": "Feedback",
                    "label": _("Feedback"),
                    "description": _("User Feedback"),

                }
            ]

        }
	]
