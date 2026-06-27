# %% [markdown]
# Introduction
#
# This project focuses on determining the Sun's rotation period by tracking
# sunspots over time. Since the sun is a hot ball of gas and plasma and not a
# regular solid surface like the Earth or a ball, the normal rules of circular
# motion do not apply here.

# %%
# Handle all the imports of the libraries here.
# %matplotlib tk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.io as ast
from sunpy.net import Fido, attrs as a
from glob import glob
from math import asin, cos, sin, pi
from datetime import datetime

# %%
"""
This section of code searches and downloads the images data from SDO and downloads and
saves them in the computer locally in the FITS file format.
"""

result = Fido.search(
    a.Time("2022/05/29", "2022/06/08"),
    a.Instrument.hmi,
    a.Physobs.intensity,
    a.Sample(ast.units.hour * 24),
)
files = Fido.fetch(result, path="../FITS Files/{file}")


# %%
def extract_metadata(filepath):
    """
    Reads the FITS file from its path and extracts the metadata (header)
    and image data into separate structures to fetch the center coordinates
    and radius of the Sun.

    Parameters:
    filepath (str): The local file path of the FITS file.

    Returns:
    tuple: A tuple containing (file_header, file_data).
    """

    with ast.fits.open(filepath) as hdulist:
        file_header = hdulist[1].header
        file_data = hdulist[1].data
        file_header["CRPIX1"]
        file_header["CRPIX2"]
        file_header["RSUN_OBS"]
        file_header["CDELT1"]
    return file_header, file_data


# %%
def generate_sunspot_coords(file_data):
    """
    Renders the 2D image array of the Sun using matplotlib.imshow.

    Note:
    This function halts execution and uses Matplotlib's ginput function,
    requiring the user to manually click on the position of the sunspot
    on the generated plot.

    Parameters:
    file_data (numpy.ndarray): The 2D array of image data from the FITS file.

    Returns:
    tuple: The (x, y) pixel coordinates of the clicked sunspot.
    """

    cmap_color = "YlOrBr_r"
    plt.imshow(file_data, cmap=cmap_color)
    sunspot_coords = plt.ginput(n=1, timeout=0)
    plt.close()
    return sunspot_coords[0]


# %%
def fetch_latitude_longitude(x, y, sun_x, sun_y, sun_R):
    """
    Converts the 2D Cartesian coordinates of the sunspot into 3D spherical
    coordinates representing latitude and longitude.

    Mathematical Formulation:
    phi = arcsin(Delta y / R)
    theta = arcsin(Delta x / (R cos(phi)))

    Parameters:
    x, y (float): The pixel coordinates of the sunspot.
    sun_x, sun_y (float): The pixel coordinates for the center of the Sun.
    sun_R (float): The radius of the Sun in pixels.

    Returns:
    tuple: (phi, theta) representing the latitude and longitude in radians.
    """

    delta_x = x - sun_x
    delta_y = sun_y - y
    phi = asin(delta_y / sun_R)  # Latitude
    theta = asin(delta_x / (sun_R * cos(phi)))  # Longitude

    return phi, theta


# %%
def calculate_synodic_period(delta_theta, delta_t):
    """
    Calculates the synodic rotation period of the Sun using the change in
    angular displacement over a given period of time.

    Mathematical Formulation:
    omega = Delta theta / Delta t
    P_syn = 2*pi / omega

    Parameters:
    delta_theta (float): Change in angular displacement (radians).
    delta_t (float): Change in time (days).

    Returns:
    float: P_syn, the synodic time period in days.
    """

    delta_theta = abs(delta_theta)
    omega = delta_theta / delta_t  # angular synodic velocity/speed of the sunspot
    P_syn = (2 * pi) / omega  # Synodic time period of the sunspot

    return P_syn


# %%
def synodic_to_sidereal(P_syn):
    """
    Converts the observed synodic time period to the true sidereal time period,
    accounting for Earth's orbital motion around the Sun.

    Mathematical Formulation:
    1/P_sid = 1/P_syn + 1/P_E

    Parameters:
    P_syn (float): The synodic time period of the sunspot in days.

    Returns:
    float: P_sid, the sidereal time period of the sunspot in days.
    """

    P_E = 365.25  # Earth's orbital period of 365.25 days
    P_sid = (P_syn * P_E) / (
        P_syn + P_E
    )  # The sidereal time of the rotation of the sun.

    return P_sid


# %%
def main():
    """
    Primary execution loop. Iterates through the local FITS files, extracts
    metadata, prompts for manual sunspot tracking, computes 3D coordinates,
    and exports the raw tracking data to a CSV.

    Returns:
    list: sunspot_data containing dictionaries of dates, coordinates, and radii.
    """

    files = sorted(glob("../FITS Files/*.FITS"))
    sunspot_data = []

    for file_path in files:
        file_header, file_data = extract_metadata(file_path)
        x_click, y_click = generate_sunspot_coords(file_data)
        date_str = file_header["DATE-OBS"]
        date_obs = datetime.fromisoformat(date_str)
        sun_center_x = file_header["CRPIX1"]
        sun_center_y = file_header["CRPIX2"]
        sun_radius = file_header["RSUN_OBS"]
        pixel_arcsecs = file_header["CDELT1"]
        sun_radius_pixels = sun_radius / pixel_arcsecs

        phi, theta = fetch_latitude_longitude(
            x_click, y_click, sun_center_x, sun_center_y, sun_radius_pixels
        )

        current_file_data = {
            "Date": date_obs,
            "Latitude": phi,
            "Longitude": theta,
            "radius": sun_radius_pixels,
        }

        sunspot_data.append(current_file_data)

    df = pd.DataFrame(sunspot_data)
    df.to_csv("sunspot_raw_data.csv", index=False)

    return sunspot_data


# %%
if __name__ == "__main__":
    main()


# %%
def differential_rotation_arrays(file_path):
    """
    Reads the exported CSV data and processes the sequential coordinates
    to calculate the sidereal angular velocities and their corresponding
    squared sine of the latitude.

    Parameters:
    file_path (str): Path to the generated CSV file containing raw sunspot data.

    Returns:
    tuple: (omegas, sin_square_phi) arrays prepared for linear regression.
    """

    omegas = []
    sin_square_phi = []

    df = pd.read_csv(file_path)
    raw_dates = df["Date"]
    dates = []
    for i in range(len((raw_dates))):
        date = pd.to_datetime(raw_dates[i])
        dates.append(date)
    phi = df["Latitude"]
    theta = df["Longitude"]
    delta_t = []
    delta_theta = []

    for i in range(len(dates) - 1):
        current_delta_t = dates[i + 1] - dates[i]
        seconds = current_delta_t.total_seconds()
        days = seconds / 86400
        current_delta_theta = abs(theta[i + 1] - theta[i])

        if days <= 0:
            continue

        days = 1
        delta_t.append(days)
        delta_theta.append(current_delta_theta)
        synodic_period = calculate_synodic_period(current_delta_theta, days)
        sidereal_period = synodic_to_sidereal(synodic_period)
        current_omega = (2 * pi) / (sidereal_period)
        omegas.append(current_omega)
        current_sin_square_phi = sin(phi[i]) ** 2
        sin_square_phi.append(current_sin_square_phi)

    return omegas, sin_square_phi


# %%
def final_graph(x_array, y_array):
    """
    Performs a linear regression on the calculated angular velocities to find
    the differential rotation constants A and B, and plots the final scatter
    graph with the line of best fit.

    Mathematical Formulation:
    omega(phi) = A + B sin^2(phi)

    Parameters:
    x_array (list): Array of sin^2(phi) values.
    y_array (list): Array of angular velocities (omega).
    """

    omegas = np.array(y_array)
    sin_square_phi = np.array(x_array)
    plt.scatter(sin_square_phi, omegas)
    plt.title("omega(phi)=A+Bsin^2(phi)")
    plt.xlabel("sin^2(phi)")
    plt.ylabel("omega(phi)")
    coeffs, uncertaintity = np.polyfit(sin_square_phi, omegas, 1, cov=True)
    slope = coeffs[0]
    intercept = coeffs[1]
    linear_fit = slope * sin_square_phi + intercept
    slope_uncer = np.sqrt(uncertaintity[0, 0])
    intercept_uncer = np.sqrt(uncertaintity[1, 1])
    B = slope * (180 / pi)
    A_uncer = intercept_uncer * (180 / pi)
    B_uncer = slope_uncer * (180 / pi)
    A = intercept * (180 / pi)
    print(
        f"The slope of the graph is B = {slope} +/- {slope_uncer} and the y-intercept is A = {intercept} +/- {intercept_uncer}."
    )
    print(f"The value of A is {A} +/- {A_uncer} degrees/day.")
    print(f"The value of B is {B} +/- {B_uncer} degrees/day.")
    plt.plot(sin_square_phi, linear_fit, color="red")
    plt.savefig("linear_regression.jpeg")
    plt.show()


# %%
diff_rot_tuples = differential_rotation_arrays("sunspot_raw_data.csv")
omegas = diff_rot_tuples[0]
sin_square_phi = diff_rot_tuples[1]
final_graph(sin_square_phi, omegas)
