from __future__ import unicode_literals

import frappe


@frappe.whitelist(allow_guest=True)
def add_feedback(name,message,subject,email):


    # doc.email=email
    #
    # doc.name=name
    # doc.message=message
    # doc.subject=subject
    # doc.insert(ignore_permissions=True,ignore_if_duplicate=True)


    print('done inserting')
    return {"resp": name}


@frappe.whitelist(allow_guest=True)
def test(name,message,subject,email):
    feedback = frappe.get_doc({
         'doctype':"Feedback",

        "name1":name,
        "email":email,
        "message":message,
        "subject":subject

    }


     )

    #
    feedback.insert(
        ignore_permissions=True,  # ignore write permissions during insert
        ignore_links=True,  # ignore Link validation in the document
        ignore_if_duplicate=True,  # dont insert if DuplicateEntryError is thrown
        ignore_mandatory=True  # insert even if mandatory fields are not set
    )
    frappe.db.commit()
