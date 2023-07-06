$(document).ready(function() {

    // Initialize dropdowns
    $('#dropdown-1').dropdown();
    $('#dropdown-2').dropdown();
    var $dropdown1 = $('#dropdown-1 .menu');
    var $dropdown2 = $('#dropdown-2 .menu');


    // Initialize the slider 1
    $('#slider-1').slider({
        min: 0,
        max: 1,
        start: 0.50,
        step: 0.01,
        onChange: function(value) {
            // Log that onChange is being triggered
            console.log('OnChange has triggered');
            console.log("Value of slider 1: " + value);

            var font_1_index = $('#dropdown-1').dropdown('get value');
            var font_2_index = $('#dropdown-2').dropdown('get value');
            var interpolation_fraction = $('#slider-1').slider('get value');

            // Log the name when dropdown is selected
            console.log("Font 1 label: " + font_1_index);
            console.log("Font 2 label: " + font_2_index);
            console.log("Interpolation fraction: " + interpolation_fraction);

            // Send POST request for interpolated images
            $.ajax({
                url: 'http://127.0.0.1:8000/interpolation',
                type: 'POST',
                data: JSON.stringify({ 
                    font_1_index: font_1_index,
                    font_2_index: font_2_index,
                    interpolation_fraction: interpolation_fraction
                }),
                contentType: "application/json; charset=utf-8",
                dataType: 'json',
                success: function(response) {
                    // Handle the response from your server
                    //console.log("Graph Response: ", JSON.stringify(response));;
                    font_1_image = response.font_1_image;
                    interpolated_image = response.interpolated_image;
                    font_2_image = response.font_2_image;

                    var img2 = document.createElement("img");
                    img2.src = interpolated_image;
                    document.getElementById('interpolated_image').appendChild(img2);
                },
                error: function (request, status, error) {
                    console.error('Error occurred:', error);
                }
    
            });
        },
    });

    // Dropdown 1
    $.getJSON('http://127.0.0.1:8000/fonts', function(fonts) {
        // Loop through each state in the returned data
        $.each(fonts, function(i, font) {
            // Append a new dropdown item for each state
            $dropdown1.append('<div class="item" data-value="' + font.value + '">' + font.name + '</div>');
        });

        // Initialize the dropdown
        $('#dropdown-1').dropdown();
    });

    // Button 2, Find Similar Fonts
    // Updates the Dropdown 2 list
    // Event handlers for buttons
    $('#btn-2').click(function() {
        
        // Log that onChange is being triggered
        console.log('Find Similar Fonts Button triggered');

        var font_index = $('#dropdown-1').dropdown('get value');

        // Log the name chosen font from the Dropdown 1
        console.log("Chosen Font label: " + font_index);


        // Here you can send the disease_name, drug_name, k1 and k2 to your server and get the response
        // Example:
        $.ajax({
            url: 'http://127.0.0.1:8000/similar_fonts',
            type: 'POST',
            data: JSON.stringify({ 
                font_index: font_index
            }),
            contentType: "application/json; charset=utf-8",
            dataType: 'json',
            success: function(similar_fonts) {
                // Clear any existing items in the dropdown
                $dropdown2.empty();
                // Loop through each state in the returned data
                $.each(similar_fonts, function(i, font) {
                    // Append a new dropdown item for each state
                    $dropdown2.append('<div class="item" data-value="' + font.value + '">' + font.name + '</div>');
                });

                // Initialize the dropdown
                $('#dropdown-2').dropdown();
            },
            error: function (request, status, error) {
                console.error('Error occurred:', error);
            }

        });
    });


    // Event handlers for buttons
    $('#btn-1').click(function() {
        
        // Log that onChange is being triggered
        console.log('Generate Button triggered');

        var font_1_index = $('#dropdown-1').dropdown('get value');
        var font_2_index = $('#dropdown-2').dropdown('get value');
        var interpolation_fraction = $('#slider-1').slider('get value');

        // Log the name when dropdown is selected
        console.log("Font 1 index: " + font_1_index);
        console.log("Font 2 index: " + font_2_index);
        console.log("Interpolation fraction: " + interpolation_fraction);

        // Send POST request for interpolated images
        $.ajax({
            url: 'http://127.0.0.1:8000/interpolation',
            type: 'POST',
            data: JSON.stringify({ 
                font_1_index: font_1_index,
                font_2_index: font_2_index,
                interpolation_fraction: interpolation_fraction
            }),
            contentType: "application/json; charset=utf-8",
            dataType: 'json',
            success: function(response) {
                // Handle the response from your server
                //console.log("Graph Response: ", JSON.stringify(response));;
                font_1_image = response.font_1_image;
                interpolated_image = response.interpolated_image;
                font_2_image = response.font_2_image;

                //var delay = 200 // delay of 200 milliseconds
                
                //var img1 = '<img src="' + font_1_image + '"/>';
                //$('#font_1_image').html(img1);

                //var img2 = '<img src="' + font_2_image + '"/>';
                //$('#font_2_image').html(img2);

                //var img3 = '<img src="' + interpolated_image + '"/>';
                //$('#interpolated_image').html(img3);


                // Check if font_1_image, interpolated_image, and font_2_image are not null before creating and appending the img elements:
                // This way, if for some reason one of the images isn't returned by the server, you won't attempt to create and append an img element with a null src attribute, which could cause an error.
                //if (font_1_image && interpolated_image && font_2_image) {

                    
                    // Clear images before populating with the new ones
                    //$('#font_1_image').empty()
                    //$('#font_2_image').empty()
                    //$('#interpolated_image').empty()

                    // Render images in id="font_1_image", "interpolation_image", "font_2_image"
                    // Create new img elements and set their src attribute
                    
                    var img1 = document.createElement("img");
                    img1.src = font_1_image;
                    document.getElementById('font_1_image').appendChild(img1);

                    var img2 = document.createElement("img");
                    img2.src = interpolated_image;
                    document.getElementById('interpolated_image').appendChild(img2);

                    var img3 = document.createElement("img");
                    img3.src = font_2_image;
                    document.getElementById('font_2_image').appendChild(img3);

                    // Render images in id="font_1_image", "interpolation_image", "font_2_image"
                    // Set the HTML content of the divs
                    //$('#font_1_image').html('<img src="' + font_1_image + '">');
                    //$('#interpolated_image').html('<img src="' + interpolated_image + '">');
                    //$('#font_2_image').html('<img src="' + font_2_image + '">');

                //}

            },
            error: function (request, status, error) {
                console.error('Error occurred:', error);
            }

        });
    });
});
