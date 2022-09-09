if (location.protocol.startsWith("http") & location.protocol !== 'https:') {
    location.replace(`https:${location.href.substring(location.protocol.length)}`);
}

window.onload = function () {
    var on_examples_page = $( ".active a:contains(Examples)" )
    if (on_examples_page.length == 1) {
        $(" .bd-sidenav li.toctree-l1:not(.has-children) ").hide();
        $(".bd-sidenav").attr('style', 'font-family: "Noto Sans" !important');
    }
};

// window.onload = function() {
//     if (window.jQuery) {
//         // jQuery is loaded
//         alert("Yeah!");
//     } else {
//         // jQuery is not loaded
//         alert("Doesn't Work");
//     }
// }
