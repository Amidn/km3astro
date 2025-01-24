"""Coordinate transformations.

Galactic:
    GC at (0, 0),
    gal. longitude, latitude (l, b)

Horizontal / altaz (km3):
    centered at detector position
    altitude, azimuth (altitude = 90deg - zenith)

EquatorialJ200 / FK5 / ICRS / GCRS
    (right ascension, declination)

    Equatorial is the same as FK5. FK5 is superseded by the ICRS, so use
    this instead. Note that FK5/ICRS are _barycentric_ implementations,
    so if you are looking for *geocentric* equatorial (i.e.
    for solar system bodies), use GCRS.


A note on maing conventions:
``phi`` and ``theta`` refer to neutrino directions, ``azimuth`` and
``zenith`` to source directions (i.e. the inversed neutrino direction).
The former says where the neutrino points to, the latter says where it comes
from.

Also radian is the default. Degree can be used, but generally the default is
to assume radian.
"""

from astropy import units as u
from astropy.units import rad, deg, hourangle  # noqa
from astropy.coordinates import (
    EarthLocation,
    SkyCoord,
    AltAz,
    Longitude,
    Latitude,
    get_sun,
    get_moon,
)
import astropy.time
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.time import Time
from astropy.coordinates import Angle

import numpy as np
import pandas as pd
import numbers

# also import get_location and convergence_angle that has been shifted to km3frame
import km3astro.frame as kf

from km3astro.constants import (
    arca_longitude,
    arca_latitude,
    arca_height,
    orca_longitude,
    orca_latitude,
    orca_height,
    antares_longitude,
    antares_latitude,
    antares_height,
)
from km3astro.time import np_to_astrotime
from km3astro.random import random_date, random_azimuth
from km3astro.sources import GALACTIC_CENTER


def neutrino_to_source_direction(phi, theta, radian=True):
    """Flip the direction.

    Parameters
    ----------
    phi, theta: neutrino direction
    radian: bool [default=True]
        receive + return angles in radian? (if false, use degree)

    """
    phi = np.atleast_1d(phi).copy()
    theta = np.atleast_1d(theta).copy()
    if not radian:
        phi *= np.pi / 180
        theta *= np.pi / 180
    assert np.all(phi <= 2 * np.pi)
    assert np.all(theta <= np.pi)
    azimuth = (phi + np.pi) % (2 * np.pi)
    zenith = np.pi - theta
    if not radian:
        azimuth *= 180 / np.pi
        zenith *= 180 / np.pi
    return azimuth, zenith


def source_to_neutrino_direction(azimuth, zenith, radian=True):
    """Flip the direction.

    Parameters
    ----------
    zenith : float
        neutrino origin
    azimuth: float
        neutrino origin
    radian: bool [default=True]
        receive + return angles in radian? (if false, use degree)

    """
    azimuth = np.atleast_1d(azimuth).copy()
    zenith = np.atleast_1d(zenith).copy()
    if not radian:
        azimuth *= np.pi / 180
        zenith *= np.pi / 180
    phi = (azimuth - np.pi) % (2 * np.pi)
    theta = np.pi - zenith
    if not radian:
        phi *= 180 / np.pi
        theta *= 180 / np.pi
    return phi, theta


def Sun(time):
    """Wrapper around astropy's get_sun, accepting numpy/pandas time objects."""
    if not isinstance(time, astropy.time.Time):
        # if np.datetime64, convert to astro time
        time = np_to_astrotime(time)
    return get_sun(time)


def Moon(time):
    """Wrapper around astropy's get_moon, accepting numpy/pandas time objects."""
    if not isinstance(time, astropy.time.Time):
        # if np.datetime64, convert to astro time
        time = np_to_astrotime(time)
    return get_moon(time)


def local_frame(time, location):
    """Get the (horizontal) coordinate frame of your detector."""
    if not isinstance(time, astropy.time.Time):
        # if np.datetime64, convert to astro time
        time = np_to_astrotime(time)
    loc = kf.get_location(location)
    frame = AltAz(obstime=time, location=loc)
    return frame


def local_event(theta, phi, time, location, radian=True, **kwargs):
    """Create astropy events from detector coordinates."""
    zenith = np.atleast_1d(theta).copy()
    azimuth = np.atleast_1d(phi).copy()

    azimuth, zenith = neutrino_to_source_direction(phi, theta, radian)

    if not radian:
        azimuth *= np.pi / 180
        zenith *= np.pi / 180
    altitude = np.pi / 2 - zenith

    loc = kf.get_location(location)
    # neutrino telescopes call the co-azimuth "azimuth"
    true_azimuth = (
        np.pi / 2 - azimuth + kf.convergence_angle(loc.lat.rad, loc.lon.rad)
    ) % (2 * np.pi)
    frame = local_frame(time, location=location)
    event = SkyCoord(alt=altitude * rad, az=true_azimuth * rad, frame=frame, **kwargs)
    return event


def sun_local(time, loc):
    """Sun position in local coordinates."""
    frame = local_frame(time, location=loc)
    sun = Sun(time)
    sun_local = sun.transform_to(frame)
    return sun_local


def moon_local(time, loc):
    """Moon position in local coordinates."""
    frame = local_frame(time, location=loc)
    moon = Moon(time)
    moon_local = moon.transform_to(frame)
    return moon_local


def gc_in_local(time, loc):
    """Galactic center position in local coordinates."""
    frame = local_frame(time, location=loc)
    gc = GALACTIC_CENTER
    gc_local = gc.transform_to(frame)
    return gc_local


class Event(object):
    def __init__(self, zenith, azimuth, time, location):
        self.zenith = zenith
        self.azimuth = azimuth
        self.time = time

    @classmethod
    def from_zenith(cls, zenith, **initargs):
        zenith = np.atleast_1d(zenith)
        n_evts = zenith.shape[0]
        azimuth = random_azimuth(n_evts)
        time = random_date(n_evts)
        return cls(zenith, azimuth, time, **initargs)


def is_args_fine_for_frame(frame, *args):

    if frame == "ParticleFrame" and len(args) < 6:
        raise TypeError(
            "Only "
            + str(len(args))
            + " given when 6 are needed: date, time, theta, phi, unit, particleframe ! for ParticleFrame"
        )

    if frame == "UTM" and len(args) < 6:
        raise TypeError(
            "Only "
            + str(len(args))
            + " given when 6 are needed: date, time, azimuth, zenith, unit, particleframe ! for UTM"
        )

    if frame == "equatorial" and len(args) < 4:
        raise TypeError(
            "Only "
            + str(len(args))
            + " given when 4 are needed: date, time, ra, dec ! for Equatorial"
        )

    if frame == "galactic" and len(args) < 4:
        raise TypeError(
            "Only "
            + str(len(args))
            + " given when 4 are needed: date, time, l, b ! for Galactic"
        )

    return 0


def build_event(Cframe, *args):
    """Build a SkyCoord object of the corresponding frame and parameters

    Parameters
    ----------
    Cframe : str
        Frame of the Skycoord event to build, either "ParticleFrame", "UTM", "equatorial" or "galactic"
    *args : list of the sky coordinate parameters

    Returns
    -------
    SkyCoord : astropy.SkyCoord
        Sky coordinate object

    """

    is_args_fine_for_frame(Cframe, *args)

    # Defining time of observation
    # using astropy.time Time because pandas.to_datetime raise an error for convertion in np_to_astrotime
    time = args[0] + "T" + args[1]
    time = Time(time)

    # ParticleFrame : date, time , theta, phi, unit, detector_name
    if Cframe == "ParticleFrame":

        theta = args[2]
        phi = args[3]
        unit = args[4]
        if unit == "deg":
            phi = phi * u.deg
            theta = theta * u.deg

        else:
            phi = phi * u.rad
            theta = theta * u.rad

        loc = kf.get_location(args[5])
        r = u.Quantity(100, u.m)  # dummy r value ! Warning !
        return SkyCoord(
            frame=kf.ParticleFrame,
            phi=phi,
            theta=theta,
            location=loc,
            obstime=time,
            r=r,
        )

    # UTM : date, time, azimuth, zenith, unit, detector
    elif Cframe == "UTM":

        az = args[2]
        zenith = args[3]
        unit = args[4]
        if unit == "deg":
            az = az * u.deg
            zenith = zenith * u.deg

        else:
            az = az * u.rad
            zenith = zenith * u.rad

        loc = kf.get_location(args[5])
        r = u.Quantity(100, u.m)  # dummy r value ! Warning !

        return SkyCoord(
            frame=kf.UTM, azimuth=az, zenith=zenith, location=loc, obstime=time, r=r
        )

    elif Cframe == "galactic":
        l = args[2]
        b = args[3]

        if isinstance(l, str):
            l = Angle(l, unit="hourangle")
        elif isinstance(l, numbers.Number):
            l = Angle(l, unit=u.deg)

        if isinstance(b, str):
            b = Angle(b, unit="hourangle")
        elif isinstance(b, numbers.Number):
            b = Angle(b, unit=u.deg)

        return SkyCoord(frame=Cframe, l=l, b=b, unit="deg", obstime=time)

    elif Cframe == "equatorial":
        ra = args[2]
        dec = args[3]

        if isinstance(ra, str):
            ra = Angle(ra, unit="hourangle")
        elif isinstance(ra, numbers.Number):
            ra = Angle(ra, unit=u.deg)

        if isinstance(dec, str):
            dec = Angle(dec, unit=u.deg)
        elif isinstance(dec, numbers.Number):
            dec = Angle(dec, unit=u.deg)

        return SkyCoord(frame=ICRS, ra=ra, dec=dec, obstime=time)

    else:
        raise ValueError("Error: Wrong Frame input:" + Cframe)
        return None


def transform_to(Skycoord, frame_to, detector_to="antares"):
    """Transform a Skycoord object to the desired frame

    Parameters
    ----------
    SkyCoord : astropy.SkyCoord
        The sky coordinate
    frame_to : str
        The desired frame of transformation, either "ParticleFrame", "UTM", "altaz", "equatorial or "galactic"
    detector_to : str [default = "antares"]
        The detector of the transformed frame, either "orca", "arca" or "antares"

    """

    time = Skycoord.obstime
    loc = kf.get_location(detector_to)

    if frame_to == "ParticleFrame":

        frame = kf.ParticleFrame(obstime=time, location=loc)
        return Skycoord.transform_to(frame)

    elif frame_to == "UTM":
        frame = kf.UTM(obstime=time, location=loc)
        return Skycoord.transform_to(frame)

    elif frame_to == "altaz":
        frame = AltAz(obstime=time, location=loc)
        return Skycoord.transform_to(frame)

    elif frame_to == "equatorial":
        return Skycoord.transform_to(ICRS)

    elif frame_to == "galactic":
        return Skycoord.transform_to("galactic")

    else:
        raise ValueError("Wrong Frame to transform: " + frame_to + " is not valid")
        return -1


def transform_to_new_frame(
    table, frame, frame_to, detector="antares", detector_to="antares"
):
    """Transform a pandas DataFrame of sky coordinate parameters to a new pandas.DataFrame of SkyCoord of the initial and desired frame

    Parameters
    ----------
    table : pandas.DataFrame(astropy.SkyCoord)
        The sky coordinate
    frame : str
        The frame of the table of parameters, either "ParticleFrame", "UTM", "altaz", "equatorial or "galactic"
    detector : str [default = "antares"]
        The detector of the sky coordinate parameters, either "orca", "arca" or "antares"
    frame_to : str
        The desired frame of transformation, either "ParticleFrame", "UTM", "altaz", "equatorial or "galactic"
    detector_to : str [default = "antares"]
        The detector of the transformed frame, either "orca", "arca" or "antares"

    """

    if frame == "ParticleFrame":
        list_evt = table.apply(
            lambda x: build_event(
                frame, x.date, x.time, x.theta, x.phi, "deg", detector
            ),
            axis=1,
            result_type="expand",
        )

    if frame == "UTM":
        list_evt = table.apply(
            lambda x: build_event(
                frame, x.date, x.time, x.azimuth, x.zenith, "deg", detector
            ),
            axis=1,
            result_type="expand",
        )

    if frame == "equatorial":
        list_evt = table.apply(
            lambda x: build_event(frame, x.date, x.time, x["RA-J2000"], x["DEC-J2000"]),
            axis=1,
            result_type="expand",
        )

    if frame == "galactic":
        list_evt = table.apply(
            lambda x: build_event(frame, x.date, x.time, x.gal_lon, x.gal_lat),
            axis=1,
            result_type="expand",
        )

    if isinstance(list_evt, pd.Series):
        series = {"SkyCoord_base": list_evt}
        list_evt = pd.DataFrame(series)

    list_evt = list_evt.set_axis(["SkyCoord_base"], axis="columns")

    list_evt["SkyCoord_new"] = list_evt.apply(
        lambda x: transform_to(x.SkyCoord_base, frame_to, detector_to),
        axis=1,
        result_type="expand",
    )

    return list_evt



class Coordinate:
    def __init__(self, dir_x, dir_y, dir_z, mjd, location=None):
        """
        Initialize the Coordinate object.

        Parameters:
            dir_x (float or array-like): x-direction cosine(s) of the event(s).
            dir_y (float or array-like): y-direction cosine(s) of the event(s).
            dir_z (float or array-like): z-direction cosine(s) of the event(s).
            mjd (float or array-like): Time(s) of the event(s) in Modified Julian Date (MJD).
            location (dict, optional): Dictionary containing latitude, longitude, and height of the detector.
                                       Defaults to KM3NeT ARCA location.
        """
        # Use KM3NeT ARCA as the default location
        if location is None:
            location = {'lat': 36.25, 'lon': 16.05, 'height': -3500}  # KM3NeT ARCA
            print("Using KM3NeT ARCA as the default location.")

        self.dir_x = np.array(dir_x)
        self.dir_y = np.array(dir_y)
        self.dir_z = np.array(dir_z)
        self.time = Time(mjd, format='mjd')  # Automatically handles arrays
        self.location = EarthLocation(lat=location['lat'], lon=location['lon'], height=location['height'])
    
    def to_altaz(self):
        """
        Convert the local Cartesian direction (dir_x, dir_y, dir_z) to AltAz coordinates (phi, theta).

        Returns:
            tuple: (phi, theta) where:
                phi (np.ndarray): Azimuth angles in degrees.
                theta (np.ndarray): Zenith angles in degrees.
        """
        # Create a SkyCoord object with Cartesian representation
        local_coord = SkyCoord(
            x=self.dir_x, y=self.dir_y, z=self.dir_z, 
            representation_type="cartesian", 
            frame=AltAz(obstime=self.time, location=self.location)
        )

        # Convert to AltAz (horizontal) coordinates
        altaz_coord = local_coord.transform_to(AltAz())

        # Calculate zenith and azimuth in degrees
        theta = 90 - altaz_coord.alt.deg  # Zenith angle (90° - Altitude)
        phi = altaz_coord.az.deg  # Azimuth angle

        return phi, theta
    
    def to_icrs(self):
        """
        Convert the local Cartesian direction to ICRS (Equatorial Coordinates).

        Returns:
            dict: Dictionary containing 'ra' (Right Ascension) and 'dec' (Declination) in degrees.
        """
        # Create a SkyCoord object with Cartesian representation
        local_coord = SkyCoord(
            x=self.dir_x, y=self.dir_y, z=self.dir_z,
            representation_type="cartesian",
            frame=AltAz(obstime=self.time, location=self.location)
        )

        # Convert from AltAz to ICRS (equatorial coordinates)
        icrs_coord = local_coord.transform_to("icrs")
        return {'ra': icrs_coord.ra.deg, 'dec': icrs_coord.dec.deg}
    
    def to_galactic(self):
        """
        Convert the local Cartesian direction to Galactic Coordinates.

        Returns:
            dict: Dictionary containing 'l' (Galactic Longitude) and 'b' (Galactic Latitude) in degrees.
        """
        # First, convert to ICRS (Equatorial Coordinates)
        icrs_coord = self.to_icrs()

        # Convert from ICRS to Galactic Coordinates
        galactic_coord = SkyCoord(ra=icrs_coord['ra'], dec=icrs_coord['dec'], unit='deg', frame="icrs").transform_to("galactic")
        return {'l': galactic_coord.l.deg, 'b': galactic_coord.b.deg}
    
    def to_fk5(self):
        """
        Convert the local Cartesian direction to FK5 (J2000 Equatorial Coordinates).

        Returns:
            dict: Dictionary containing 'ra' (Right Ascension) and 'dec' (Declination) in degrees.
        """
        # Create a SkyCoord object with Cartesian representation
        local_coord = SkyCoord(
            x=self.dir_x, y=self.dir_y, z=self.dir_z,
            representation_type="cartesian",
            frame=AltAz(obstime=self.time, location=self.location)
        )

        # Convert from AltAz to FK5 (J2000)
        fk5_coord = local_coord.transform_to('fk5')
        return {'ra': fk5_coord.ra.deg, 'dec': fk5_coord.dec.deg}
