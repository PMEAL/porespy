// We don't want the sidebar to include list of examples, that's just
// much. This simple script tries to select those items and hide them.
// "aside" selects sidebar elements, and "href" narrows it down to the
// list of examples. This is a workaround, not a permanent fix.

var examples = $('aside a[href*="examples.html"]')
var examples_clicked = $( ":contains('Examples')" ).filter($( ".current.reference.internal" ))

examples.nextAll().hide()
examples_clicked.nextAll().hide()
