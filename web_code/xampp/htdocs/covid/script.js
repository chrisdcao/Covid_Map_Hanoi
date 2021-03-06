setInterval(refreshMap, 3000);

var mymap = L.map('mapid').setView([21.022736, 105.8019441], 13);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    id: 'vietnam/streets-2019'
}).addTo(mymap);

// parent class for icons (vì các instance có nhiều điểm tương đồng)
var CovidIcon = L.Icon.extend({
    options: {
        iconSize:     [26, 26],
        iconAnchor:   [16, 25],
        popupAnchor:  [-2, -17]
    }
});
// các instance để sau sử dụng
var redMarker    = new CovidIcon({iconUrl: './img/covid_red.png'});
var greenMarker  = new CovidIcon({iconUrl: './img/marker_green.png'});
var yellowMarker = new CovidIcon({iconUrl: './img/marker4.png'});

// create marker
// var marker = L.marker([51.5, -0.09], {icon: redMarker}).addTo(mymap);
// the string here must be pre-formatted in python first, then push up onto the database. An example form:
// 1. <b>Mã bệnh nhân:</b><br><b>Địa chỉ:</b>23 dduwofng....<br><b>Mức độ:</b>Nguy hiểm<br><b>Thời gian tới An toàn còn:</b>21 ngày
// marker.bindPopup("<b>Hello world!</b><br>I am a popup.").openPopup();

var locations = []

// stand alone popup
var popup = L.popup()
    .setLatLng([21.0357145, 105.8038324])
    .setContent("Welcome to Hanoi Covid Map!")
    .openOn(mymap);

var popup_uninit = L.popup();

// event interaction
//+ e.latlng.toString()
function onMapClick(e) {
    popup_uninit
        .setLatLng(e.latlng)
        .setContent("No Covid Here!")
        .openOn(mymap);
}

mymap.on('click', onMapClick);

downloadUrl('getmarkers.php', function(data) {
    var xml = data.responseXML;
    var markers = xml.documentElement.getElementsByTagName('marker');

    Array.prototype.forEach.call(markers, function(markerElem) {
        // var id = markerElem.getAttribute('id');
        var name = markerElem.getAttribute('name').split("%");
        var address = markerElem.getAttribute('address').split("%");
        var subject = markerElem.getAttribute('subject').split("%");
        var type = markerElem.getAttribute('type');
        var lat = markerElem.getAttribute('lat');
        var lng = markerElem.getAttribute('lng');
        if ( type != 'None' ) {
            if (type == 'yellow') {
                var myMarker = L.marker([lat, lng], {icon: yellowMarker}).addTo(mymap);
            } else if ( type == 'green' ) {
                var myMarker = L.marker([lat, lng], {icon: greenMarker}).addTo(mymap);
            } else {
                var myMarker = L.marker([lat, lng], {icon: redMarker}).addTo(mymap);
            }

            myMarker.bindPopup("<b>" + name[0] + "</b>" + " " + name[1] + "<br>" +
                               "<b>" + address[0] + "</b>" + " " + address[1] + "<br>" +
                               "<b>" + subject[0] + "</b>" + " " + subject[1] + "<br>"
                              ).openPopup();
        }
    });
});

function refreshMap() {
    downloadUrl('getnewmarkers.php', function(data) {
        var xml = data.responseXML;
        var markers = xml.documentElement.getElementsByTagName('marker');

        Array.prototype.forEach.call(markers, function(markerElem) {
            //var id = markerElem.getAttribute('id');
            var name = markerElem.getAttribute('name');
            var address = markerElem.getAttribute('address');
            var subject = markerElem.getAttribute('subject');
            var type = markerElem.getAttribute('type');
            var lat = markerElem.getAttribute('lat');
            var lng = markerElem.getAttribute('lng');
            if ( type != 'None' ) {
                if (type == 'yellow') {
                    var myMarker = L.marker([lat, lng], {icon: yellowMarker}).addTo(mymap);
                } else if ( type == 'green' ) {
                    var myMarker = L.marker([lat, lng], {icon: greenMarker}).addTo(mymap);
                } else {
                    var myMarker = L.marker([lat, lng], {icon: redMarker}).addTo(mymap);
                }

                myMarker.bindPopup("<b>" + name[0] + "</b>" + " " + name[1] + "<br>" +
                                   "<b>" + address[0] + "</b>" + " " + address[1] + "<br>" +
                                   "<b>" + subject[0] + "</b>" + " " + subject[1] + "<br>"
                                  ).openPopup();
            }
        });
    });
}

function downloadUrl(url, callback) {
    var request = window.ActiveXObject ?
    new ActiveXObject('Microsoft.XMLHTTP') :
    new XMLHttpRequest;

    request.onreadystatechange = function() {
        if (request.readyState == 4) {
        request.onreadystatechange = doNothing;
        callback(request, request.status);
        }
    };

    request.open('GET', url, true);
    request.send(null);
}

function doNothing() {}

