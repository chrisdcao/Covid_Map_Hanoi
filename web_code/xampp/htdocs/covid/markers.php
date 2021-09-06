<?php
require("settings.php");

// Gets data from URL parameters.
$name = $_GET['name'];
$address = $_GET['address'];
$subject = $_GET['subject'];
$lat = $_GET['lat'];
$lng = $_GET['lng'];
$type = $_GET['type'];

// Opens a connection to a mysqli server.
$connection=mysqli_connect ($servername, $username, $password);
if (!$connection) {
  die('Not connected : ' . mysqli_error());
}

// Sets the active mysqli database.
$db_selected = mysqli_select_db($database, $connection);
if (!$db_selected) {
  die ('Can\'t use db : ' . mysqli_error());
}

// Inserts new row with place data.
$query = sprintf("INSERT INTO markers " .
         " (id, name, address, subject, lat, lng, type ) " .
         " VALUES (NULL, '%s', '%s', '%s', '%s', '%s', '%s');",
         mysqli_real_escape_string($name),
         mysqli_real_escape_string($address),
         mysqli_real_escape_string($subject),
         mysqli_real_escape_string($lat),
         mysqli_real_escape_string($lng),
         mysqli_real_escape_string($type));

$result = mysqli_query($query);

if (!$result) {
  die('Invalid query: ' . mysqli_error());
}

?>
