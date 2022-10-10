$(document).ready(function () {
    $(".pop").each(setPopover);

    $("#language").on("change", loadErrorChannels);

    $("#sentence").on("keyup", setCounters);

    $("#btn-edit").on("click", edit);

    $("#btn-copy").on("click", copy);

    $("#btn-clear").on("click", clear);

    Initilize();
});

function Initilize() {
    $("#sentence").keyup();
}

function setPopover() {
    new bootstrap.Popover(this, {
        trigger: "focus",
        html: true
    });
}

function loadErrorChannels() {
    $.get("/get_error_channels", {
        language: $("#language").val()
    }, function (data) {
        $("#error-channel").empty();
        var $options = [];
        $options.push($("<option></option>").text("Seleccionar"));
        $.each(data, function (i, item) {
            $options.push($("<option value='" + item.value + "'>" + item.text + "</option>"));
        });
        $("#error-channel").append($options);
    })

    return false;
}

function setCounters() {
    var sentence = $(this).val().trim();

    $("#characters-count").text(sentence.length);
    $("#words-count").text(sentence ? sentence.split(' ').length : 0);
}

function edit() {
    editableMode();
}

function editableMode() {
    $("#sentence").removeClass("d-none");
    $("#box").addClass("d-none");
    $(".no-editable-mode-button-group").addClass("d-none");
    $(".editable-mode-button-group").removeClass("d-none");
}

function copy() {
    var $temp = $("<input>");
    $("body").append($temp);
    var text = $("#sentence").text();
    $temp.val(text).select();
    document.execCommand("copy");
    $temp.remove();
}

function clear() {
    $("#sentence").text("").keyup();
    editableMode();
}